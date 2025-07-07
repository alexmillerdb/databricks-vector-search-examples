# Databricks notebook source
# MAGIC %md ## Option 1: Async Vector Search with Configurable Concurrency using `asyncio` and `httpx`
# MAGIC - Loads the Spark dataframe, selects the `query_text` (can also provide `query_vector` for embeddings), and converts to Python list
# MAGIC - This will load data into memory and then runs async processes with automatic retry logic
# MAGIC - Good for datasets < 1M records
# MAGIC - Can use Serverless CPU compute

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch httpx
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# VS Endpoint Name
VECTOR_SEARCH_ENDPOINT = "abs_test_temp"
# Index-Name
VECTOR_SEARCH_INDEX = "users.alex_miller.spark_docs_vs_index"

# Embedding Dimensions
EMBEDDING_DIMENSION = 1024

# Text Dataset having embeddings
SOURCE_DATASET = "users.alex_miller.spark_docs_gold"

source_df = spark.table(SOURCE_DATASET)

# COMMAND ----------

source_df.count()

# COMMAND ----------

# MAGIC %md ### Single REST API Call

# COMMAND ----------

# single example text query
single_row_example = source_df.select("content", "uuid").limit(1).toPandas()
query_text = single_row_example['content'][0]
uuid = single_row_example['uuid'][0]

print("Query Text: ", query_text)
print("UUID: ", uuid)

# COMMAND ----------

import requests
import json

# Configuration
WORKSPACE_URL = dbutils.entry_point.getDbutils().notebook().getContext().apiUrl().get()
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
INDEX_NAME = "users.alex_miller.spark_docs_vs_index"
columns_to_include = ["filepath", "content", "category", "uuid"]

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

# # Create the filters dictionary
# filters_dict = {"uuid": uuid}

# # Serialize to JSON string as required by the API
# filters_json = json.dumps(filters_dict)

payload = {
    "num_results": 5,
    "query_text": query_text,
    "columns": columns_to_include,
    "query_type": "HYBRID", # Or ANN,
    # "filters_json": filters_json # Not supported in storage optimized endpoints
}

response = requests.post(
    f"{WORKSPACE_URL}/api/2.0/vector-search/indexes/{INDEX_NAME}/query",
    headers=headers,
    json=payload
)

data = response.json()

# Get column names and their indices
columns = [col['name'] for col in data['manifest']['columns']]
col_idx = {name: idx for idx, name in enumerate(columns)}

# Prepare the result dictionary
result_dict = {
    "filepath": [],
    "uuid": [],
    "content": [],
    "score": []
}

# Extract rows and populate the dictionary
for row in data['result']['data_array']:
    result_dict['filepath'].append(row[col_idx['filepath']])
    result_dict['uuid'].append(row[col_idx['uuid']])
    result_dict['content'].append(row[col_idx['content']])
    result_dict['score'].append(row[col_idx['score']])

# result_dict now has the desired structure
print(result_dict)


# COMMAND ----------

# MAGIC %md ### Async functions

# COMMAND ----------

import json
import asyncio
import httpx
from typing import List, Optional

def get_config(index_name=None):
    """Retrieve Databricks workspace URL, API token, and index name."""
    WORKSPACE_URL = dbutils.entry_point.getDbutils().notebook().getContext().apiUrl().get()
    TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    INDEX_NAME = index_name or "users.alex_miller.spark_docs_vs_index"
    return WORKSPACE_URL, TOKEN, INDEX_NAME

def build_headers(token):
    """Build HTTP headers for the API call."""
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

def build_payload(
    query_text=None,
    query_vector=None,
    columns=None,
    num_results=5,
    query_type="HYBRID",
    filters_json=None,
    **kwargs
):
    """Construct the payload for the vector search query."""
    payload = {
        "num_results": num_results,
        "columns": columns or ["filepath", "content", "category", "uuid"],
        "query_type": query_type
    }
    if query_text is not None:
        payload["query_text"] = query_text
    if query_vector is not None:
        payload["query_vector"] = query_vector
    if filters_json is not None:
        payload["filters_json"] = filters_json
    payload.update(kwargs)
    return payload

async def query_vector_search(
    client, workspace_url, index_name, headers, payload,
    max_retries=5, backoff_factor=2
):
    """Async API call with retry logic for transient errors."""
    url = f"{workspace_url}/api/2.0/vector-search/indexes/{index_name}/query"
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429 or response.status_code >= 500:
                wait = backoff_factor ** attempt
                print(f"Retrying in {wait} seconds due to status {response.status_code}...")
                await asyncio.sleep(wait)
            else:
                print(f"Failed: {response.status_code} - {response.text}")
                break
        except httpx.RequestError as e:
            print(f"Request error: {e}, retrying...")
            await asyncio.sleep(backoff_factor ** attempt)
    return {}

def parse_response(data, fields=None):
    """Parse API response to extract desired fields as list of row dicts."""
    if not data or "result" not in data or "data_array" not in data["result"]:
        return []
    columns = [col['name'] for col in data['manifest']['columns']]
    col_idx = {name: idx for idx, name in enumerate(columns)}
    fields = fields or ["filepath", "uuid", "content", "score"]
    rows = []
    for row in data['result']['data_array']:
        row_dict = {field: row[col_idx[field]] for field in fields if field in col_idx}
        rows.append(row_dict)
    return rows

async def async_vector_search_batch(
    queries: List[str],
    index_name: Optional[str] = None,
    columns: Optional[List[str]] = None,
    num_results: int = 5,
    query_type: str = "HYBRID",
    filters_json: Optional[str] = None,
    query_vector_list: Optional[List[list]] = None,
    lookup_ids: Optional[List[str]] = None,
    concurrency: int = 100,
    **kwargs
):
    """
    Run vector search for a batch of queries asynchronously with concurrency control.
    Optionally attaches lookup_content and lookup_id to each result row.
    """
    workspace_url, token, index_name = get_config(index_name)
    headers = build_headers(token)
    columns = columns or ["filepath", "content", "category", "uuid"]

    semaphore = asyncio.Semaphore(concurrency)

    async def sem_task(query, idx):
        async with semaphore:
            payload = build_payload(
                query_text=query if query_vector_list is None else None,
                query_vector=None if query_vector_list is None else query_vector_list[idx],
                columns=columns,
                num_results=num_results,
                query_type=query_type,
                filters_json=filters_json,
                **kwargs
            )
            async with httpx.AsyncClient(timeout=30) as client:
                response = await query_vector_search(client, workspace_url, index_name, headers, payload)
            parsed_results = parse_response(response, fields=columns + ["score"])
            # Attach lookup_content and lookup_id if provided
            for row in parsed_results:
                row["lookup_content"] = query
                if lookup_ids is not None:
                    row["lookup_id"] = lookup_ids[idx]
            return parsed_results

    tasks = [sem_task(query, i) for i, query in enumerate(queries)]
    results = await asyncio.gather(*tasks)
    all_rows = [row for batch in results for row in batch]
    return all_rows


# COMMAND ----------

# MAGIC %md ### Run Async functions with 100 concurrency

# COMMAND ----------

# Extract query texts from the DataFrame
query_texts = source_df.select("content").limit(1000).toPandas()["content"].tolist()
lookup_ids = source_df.select("uuid").limit(1000).toPandas()["uuid"].tolist()
print(f"Prepared {len(query_texts)} queries for vector search.")

# Run async batch search with controlled concurrency (e.g., 100)
async def run_async_batch():
    return await async_vector_search_batch(
        queries=query_texts,
        lookup_ids=lookup_ids,
        index_name="users.alex_miller.spark_docs_vs_index",
        columns=["filepath", "content", "category", "uuid"],
        num_results=5,
        query_type="HYBRID",
        concurrency=100
    )

# Execute the async function
all_rows = asyncio.run(run_async_batch())

# Create and display the Spark DataFrame
sdf = spark.createDataFrame(all_rows)
print(f"Spark DataFrame created with {sdf.count()} rows.")
display(sdf)

# COMMAND ----------

# MAGIC %md ## Option 2: Use Ray (Ray Data and Ray Core) to load data into Ray dataset and run async processing
# MAGIC - Cluster config below but make sure to update `setup_ray_cluster` configurations to correct setting ([documentation](https://docs.databricks.com/aws/en/machine-learning/ray/scale-ray))
# MAGIC - Utilizes similar processing logic as above but uses Ray Data and Ray Core to orchestrate the parallel processing
# MAGIC - Good for large datasets that don't fit into memory (why we use Ray Data)

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch httpx
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster
import ray

# Configure Ray cluster (adjust worker nodes and CPUs as needed)
setup_ray_cluster(
  min_worker_nodes=1,     # Minimum number of worker nodes on Spark cluster
  max_worker_nodes=5,     # Maximum number of worker nodes on Spark cluster
  num_cpus_per_node=16,   # Number of CPUs per worker node
  num_cpus_head_node=8,   # Number of CPUs on head node (give Spark some CPUs)
  num_gpus_head_node=0,
  num_gpus_worker_node=0
  )
ray.init(ignore_reinit_error=True)

# COMMAND ----------

# MAGIC %md ### Data Prep and configuration

# COMMAND ----------

import ray.data

# VS Endpoint Name
VECTOR_SEARCH_ENDPOINT = "abs_test_temp"
# Index-Name
VECTOR_SEARCH_INDEX = "users.alex_miller.spark_docs_vs_index"
# UC variables
CATALOG = "users"
SCHEMA = "alex_miller"
VOLUME_NAME = "ray"
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME_NAME}")

# Embedding Dimensions
EMBEDDING_DIMENSION = 1024

# Text Dataset having embeddings
SOURCE_DATASET = f"{CATALOG}.{SCHEMA}.spark_docs_gold"
UC_VOLUME_FOR_RAY = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/temp"   # temp directory in UC to store Ray dataset creation
VECTOR_COLUMN = "content" # use embeddings for vector_query
VECTOR_ID = "uuid"

# Convert a Spark DataFrame to a Ray Dataset
spark_df = spark.read.table(SOURCE_DATASET).select([VECTOR_COLUMN, VECTOR_ID]).limit(1000)  # Or use your DataFrame source
ray_ds = ray.data.from_spark(spark_df)

# COMMAND ----------

ray_ds.take_batch(2)

# COMMAND ----------

import ray
import httpx
import json

@ray.remote
def vector_search_task(query, workspace_url, index_name, token, columns, num_results, query_type, filters_json):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "num_results": num_results,
        "columns": columns,
        "query_type": query_type,
        "query_text": query,
    }
    if filters_json:
        payload["filters_json"] = filters_json

    url = f"{workspace_url}/api/2.0/vector-search/indexes/{index_name}/query"
    for attempt in range(1, 6):
        try:
            with httpx.Client(timeout=30) as client:
                response = client.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                # Parse result into a row dict
                columns_list = [col['name'] for col in data['manifest']['columns']]
                col_idx = {name: idx for idx, name in enumerate(columns_list)}
                rows = []
                for row in data['result']['data_array']:
                    row_dict = {field: row[col_idx[field]] for field in columns + ["score"] if field in col_idx}
                    rows.append(row_dict)
                return rows
            elif response.status_code == 429 or response.status_code >= 500:
                import time
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                break
        except Exception as e:
            import time
            time.sleep(2 ** attempt)
    return []

def flatten_results(results, columns, lookup_values, lookup_ids):
    """
    Flattens nested results, extracts specified columns plus 'score',
    and attaches the original lookup value and ID for each query.
    """
    if 'score' not in columns:
        columns = columns + ['score']
    all_rows = []
    for lookup_value, lookup_id, result_list in zip(lookup_values, lookup_ids, results):
        for result in result_list:
            row = {col: result.get(col) for col in columns}
            row["lookup_content"] = lookup_value
            row["lookup_id"] = lookup_id
            all_rows.append(row)
    return all_rows

def ray_vector_search_batch(ray_ds, workspace_url, index_name, token, columns, num_results, query_type, filters_json, batch_size=50):
    def process_batch(batch):
        # batch is a dict of arrays: batch[VECTOR_COLUMN], batch[VECTOR_ID]
        contents = batch[VECTOR_COLUMN]
        ids = batch[VECTOR_ID]
        results = ray.get([
            vector_search_task.remote(
                content, workspace_url, index_name, token, columns, num_results, query_type, filters_json
            ) for content in contents
        ])
        all_rows = flatten_results(results, columns, contents, ids)
        output_columns = columns + (["score"] if "score" not in columns else []) + ["lookup_content", "lookup_id"]
        output_dict = {col: [row.get(col) for row in all_rows] for col in output_columns}
        return output_dict

    results = ray_ds.map_batches(process_batch, batch_size=batch_size)
    return results

# COMMAND ----------

# Configuration
WORKSPACE_URL = dbutils.entry_point.getDbutils().notebook().getContext().apiUrl().get()
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
INDEX_NAME = "users.alex_miller.spark_docs_vs_index"
COLUMNS = ["filepath", "content", "category", "uuid"]
NUM_RESULTS = 5
QUERY_TYPE = "HYBRID"
FILTERS_JSON = None  # Or your filter as a JSON string

# Run distributed vector search and flatten results
all_rows = ray_vector_search_batch(
    ray_ds=ray_ds,
    workspace_url=WORKSPACE_URL,
    index_name=INDEX_NAME,
    token=TOKEN,
    columns=COLUMNS,
    num_results=NUM_RESULTS,
    query_type=QUERY_TYPE,
    filters_json=FILTERS_JSON,
    batch_size=50
)

# write Ray Data to Spark
import os
dbutils.fs.mkdirs(UC_VOLUME_FOR_RAY)
os.environ['RAY_UC_VOLUMES_FUSE_TEMP_DIR'] = UC_VOLUME_FOR_RAY
_ = ray.data.Dataset.write_databricks_table(
  ray_dataset=all_rows,
  name="users.alex_miller.spark_docs_vs_batch_results",
  mode="overwrite"    # or append
)

shutdown_ray_cluster()
ray.shutdown()

# Read and display a Spark DataFrame
sdf = spark.read.table("users.alex_miller.spark_docs_vs_batch_results")
display(sdf)
