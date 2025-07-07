# Databricks notebook source
# MAGIC %md ## Ray Vector Search with Distributed Processing using Ray Data and Ray Core
# MAGIC - Cluster config below but make sure to update `setup_ray_cluster` configurations to correct setting ([documentation](https://docs.databricks.com/aws/en/machine-learning/ray/scale-ray))
# MAGIC - Utilizes Ray Data and Ray Core to orchestrate the parallel processing
# MAGIC - Good for large datasets that don't fit into memory (why we use Ray Data)
# MAGIC - Scales processing across multiple worker nodes

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch httpx ray[default]
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

# MAGIC %md ### Data Preparation and Configuration

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

# Check Ray dataset structure
ray_ds.take_batch(2)

# COMMAND ----------

# MAGIC %md ### Ray Remote Functions and Processing Logic

# COMMAND ----------

import ray
import httpx
import json

@ray.remote
def vector_search_task(query, workspace_url, index_name, token, columns, num_results, query_type, filters_json):
    """Ray remote function to perform vector search for a single query."""
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
    """
    Process Ray dataset in batches using Ray remote functions for vector search.
    """
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

# MAGIC %md ### Execute Ray Vector Search Processing

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

print("Ray processing completed successfully!")

# COMMAND ----------

# MAGIC %md ### Save Results to Delta Table

# COMMAND ----------

# Write Ray Data to Spark/Delta table
import os
dbutils.fs.mkdirs(UC_VOLUME_FOR_RAY)
os.environ['RAY_UC_VOLUMES_FUSE_TEMP_DIR'] = UC_VOLUME_FOR_RAY

# Write to Delta table
_ = ray.data.Dataset.write_databricks_table(
  ray_dataset=all_rows,
  name="users.alex_miller.spark_docs_vs_ray_results",
  mode="overwrite"    # or append
)

print("Results written to Delta table: users.alex_miller.spark_docs_vs_ray_results")

# COMMAND ----------

# MAGIC %md ### Read and Display Results

# COMMAND ----------

# Read and display the results as a Spark DataFrame
sdf = spark.read.table("users.alex_miller.spark_docs_vs_ray_results")
print(f"Spark DataFrame created with {sdf.count()} rows.")
display(sdf)

# COMMAND ----------

# MAGIC %md ### Cleanup Ray Resources

# COMMAND ----------

# Clean up Ray cluster
shutdown_ray_cluster()
ray.shutdown()

print("Ray cluster shutdown completed.")

# COMMAND ----------

# MAGIC %md ### Optional: Performance Comparison
# MAGIC 
# MAGIC This Ray approach is particularly beneficial for:
# MAGIC - Large datasets that don't fit into memory
# MAGIC - Distributed processing across multiple nodes
# MAGIC - Complex data transformations combined with vector search
# MAGIC - Scenarios where you need to process data in streaming fashion 