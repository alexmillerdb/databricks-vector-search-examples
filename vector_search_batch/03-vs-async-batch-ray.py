# Databricks notebook source
# MAGIC %md ## Ray Vector Search with Distributed Processing using Ray Data and Ray Core
# MAGIC - Cluster config below but make sure to update `setup_ray_cluster` configurations to correct setting ([documentation](https://docs.databricks.com/aws/api/databricks-apps/ray/scale-ray))
# MAGIC - Utilizes Ray Data and Ray Core to orchestrate the parallel processing
# MAGIC - Supports **ANN (vector-only)** and **HYBRID (text+vector)** search modes
# MAGIC - Good for large datasets that don't fit into memory (why we use Ray Data)
# MAGIC - Scales processing across multiple worker nodes
# MAGIC - **Compatible with IMDB embeddings dataset from 02-create-vector-search-index.ipynb**

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

# UC CATALOG, SCHEMA, INDEX (matching 02-create-vector-search-index.ipynb)
UC_CATALOG = "users"
UC_SCHEMA = "alex_miller"
VS_INDEX_NAME = "vs_batch_example"

# VS Endpoint Name
VECTOR_SEARCH_ENDPOINT = "abs_test_temp"
# Index-Name
VECTOR_SEARCH_INDEX = f"{UC_CATALOG}.{UC_SCHEMA}.{VS_INDEX_NAME}"

# Embedding Dimensions
EMBEDDING_DIMENSION = 1024

# Text Dataset having embeddings (matching the source table from index creation)
SOURCE_DATASET = f"{UC_CATALOG}.{UC_SCHEMA}.imdb_embeddings"

# ID and Content Columns
ID_COLUMN = "id"
EMBEDDINGS_COLUMN = "embeddings"
TEXT_COLUMN = "text"

# UC Volume for Ray processing
VOLUME_NAME = "ray"
spark.sql(f"CREATE VOLUME IF NOT EXISTS {UC_CATALOG}.{UC_SCHEMA}.{VOLUME_NAME}")
UC_VOLUME_FOR_RAY = f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/{VOLUME_NAME}/temp"

# Convert a Spark DataFrame to a Ray Dataset
spark_df = spark.read.table(SOURCE_DATASET).select([ID_COLUMN, EMBEDDINGS_COLUMN, TEXT_COLUMN]).limit(1000)
ray_ds = ray.data.from_spark(spark_df)

# COMMAND ----------

# Check Ray dataset structure
ray_ds.take_batch(2)

# COMMAND ----------

# MAGIC %md ### Helper Functions and Payload Building

# COMMAND ----------

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
        "columns": columns or [ID_COLUMN, EMBEDDINGS_COLUMN, TEXT_COLUMN],
        "query_type": query_type
    }
    
    # Handle query parameters based on query_type
    if query_type == "ANN":
        # ANN (vector-only): Only add query_vector, never query_text
        if query_vector is not None:
            try:
                # Ensure vector is a list and has proper dimensions
                if hasattr(query_vector, 'tolist'):
                    vec_list = query_vector.tolist()
                elif isinstance(query_vector, list):
                    vec_list = query_vector
                else:
                    vec_list = list(query_vector)
                
                # Only add if vector has content
                if len(vec_list) > 0:
                    payload["query_vector"] = vec_list
            except Exception as e:
                print(f"Error processing query_vector: {e}")
                # Skip vector if there's an issue
                
    elif query_type == "HYBRID":
        # HYBRID: Can use both query_text and query_vector
        if query_text is not None and query_text != "":
            payload["query_text"] = query_text
        
        if query_vector is not None:
            try:
                # Ensure vector is a list and has proper dimensions
                if hasattr(query_vector, 'tolist'):
                    vec_list = query_vector.tolist()
                elif isinstance(query_vector, list):
                    vec_list = query_vector
                else:
                    vec_list = list(query_vector)
                
                # Only add if vector has content
                if len(vec_list) > 0:
                    payload["query_vector"] = vec_list
            except Exception as e:
                print(f"Error processing query_vector: {e}")
                # Skip vector if there's an issue
                
    else:
        # Text-only or other query types: Only add query_text
        if query_text is not None and query_text != "":
            payload["query_text"] = query_text
    
    # Add filters if provided
    if filters_json is not None:
        payload["filters_json"] = filters_json
    
    # Add any additional kwargs
    payload.update(kwargs)
    return payload

# COMMAND ----------

# MAGIC %md ### Ray Remote Functions and Processing Logic

# COMMAND ----------

import ray
import httpx
import json

@ray.remote
def vector_search_task(query_text, query_vector, workspace_url, index_name, token, columns, num_results, query_type, filters_json):
    """Ray remote function to perform vector search for a single query with support for text, vector, or hybrid queries."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Build payload using the same logic as the Python version
    payload = build_payload(
        query_text=query_text,
        query_vector=query_vector,
        columns=columns,
        num_results=num_results,
        query_type=query_type,
        filters_json=filters_json
    )

    url = f"{workspace_url}/api/2.0/vector-search/indexes/{index_name}/query"
    
    # Retry logic with exponential backoff
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
                print(f"Retrying in {wait} seconds due to status {response.status_code}...")
                time.sleep(wait)
            else:
                print(f"Failed: {response.status_code} - {response.text}")
                break
        except Exception as e:
            import time
            print(f"Request error: {e}, retrying...")
            time.sleep(2 ** attempt)
    return []

def flatten_results(results, columns, lookup_texts, lookup_vectors, lookup_ids):
    """
    Flattens nested results, extracts specified columns plus 'score',
    and attaches the original lookup values and ID for each query.
    """
    if 'score' not in columns:
        columns = columns + ['score']
    all_rows = []
    for i, (result_list, lookup_id) in enumerate(zip(results, lookup_ids)):
        lookup_text = lookup_texts[i] if i < len(lookup_texts) else None
        lookup_vector = lookup_vectors[i] if i < len(lookup_vectors) else None
        
        for result in result_list:
            row = {col: result.get(col) for col in columns}
            # Add lookup information
            if lookup_text is not None:
                row["lookup_content"] = lookup_text
            if lookup_vector is not None:
                row["lookup_embeddings"] = lookup_vector
            row["lookup_id"] = lookup_id
            all_rows.append(row)
    return all_rows

def ray_vector_search_batch(
    ray_ds, 
    workspace_url, 
    index_name, 
    token, 
    columns, 
    num_results, 
    query_type, 
    filters_json, 
    batch_size=50
):
    """
    Process Ray dataset in batches using Ray remote functions for vector search.
    Supports ANN andHYBRID search modes.
    """
    def process_batch(batch):
        # Extract data from batch
        texts = batch.get(TEXT_COLUMN, [])
        vectors = batch.get(EMBEDDINGS_COLUMN, [])
        ids = batch[ID_COLUMN]
        
        # Convert vectors to proper format
        processed_vectors = []
        for vec in vectors:
            if vec is not None:
                if hasattr(vec, 'tolist'):
                    processed_vectors.append(vec.tolist())
                elif isinstance(vec, list):
                    processed_vectors.append(vec)
                else:
                    processed_vectors.append(list(vec))
            else:
                processed_vectors.append(None)
        
        # Prepare query parameters based on query_type
        query_texts = []
        query_vectors = []
        
        if query_type == "ANN":
            # ANN: Only use vectors, set text to None
            query_texts = [None] * len(ids)
            query_vectors = processed_vectors
        elif query_type == "HYBRID":
            # HYBRID: Use both text and vectors
            query_texts = texts
            query_vectors = processed_vectors
        else:
            # Text-only: Only use text, set vectors to None
            query_texts = texts
            query_vectors = [None] * len(ids)
        
        # Execute vector search tasks
        results = ray.get([
            vector_search_task.remote(
                text, vector, workspace_url, index_name, token, columns, num_results, query_type, filters_json
            ) for text, vector in zip(query_texts, query_vectors)
        ])
        
        # Flatten results and add lookup information
        all_rows = flatten_results(results, columns, texts, processed_vectors, ids)
        
        # Prepare output columns
        output_columns = columns + (["score"] if "score" not in columns else [])
        if query_type != "ANN":
            output_columns.append("lookup_content")
        if query_type != "text-only":
            output_columns.append("lookup_embeddings")
        output_columns.append("lookup_id")
        
        # Convert to output format
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
INDEX_NAME = VECTOR_SEARCH_INDEX
COLUMNS = [ID_COLUMN, EMBEDDINGS_COLUMN, TEXT_COLUMN]
NUM_RESULTS = 5
QUERY_TYPE = "ANN"  # Options: "ANN", "HYBRID"
FILTERS_JSON = None  # Or your filter as a JSON string

print("=== Configuration ===")
print(f"Index: {INDEX_NAME}")
print(f"Query Type: {QUERY_TYPE}")
print(f"Columns: {COLUMNS}")
print(f"Number of results per query: {NUM_RESULTS}")

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
os.environ['RAY_USE_LEGACY_RAYLET'] = '1'  # Sometimes needed for compatibility
os.environ['RAY_UC_VOLUMES_FUSE_TEMP_DIR'] = UC_VOLUME_FOR_RAY

# Write to Delta table
_ = ray.data.Dataset.write_databricks_table(
  ray_dataset=all_rows,
  name=f"{UC_CATALOG}.{UC_SCHEMA}.imdb_vs_ray_results",
  mode="overwrite"    # or append
)

print(f"Results written to Delta table: {UC_CATALOG}.{UC_SCHEMA}.imdb_vs_ray_results")

# COMMAND ----------

# MAGIC %md ### Read and Display Results

# COMMAND ----------

# Read and display the results as a Spark DataFrame
sdf = spark.read.table(f"{UC_CATALOG}.{UC_SCHEMA}.imdb_vs_ray_results")
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

# MAGIC %md ### Configuration Summary and Usage Guide
# MAGIC 
# MAGIC This notebook has been updated to work with:
# MAGIC - **Vector Search Index**: `users.alex_miller.vs_batch_example`
# MAGIC - **Source Dataset**: `users.alex_miller.imdb_embeddings`
# MAGIC - **Endpoint**: `abs_test_temp`
# MAGIC - **Expected Columns**: `id`, `embeddings`, `text`
# MAGIC 
# MAGIC ### Query Type Options & API Behavior:
# MAGIC 
# MAGIC #### **ANN (Vector-Only Search)**
# MAGIC - **API Requirement**: Only `query_vector` allowed, `query_text` must be None
# MAGIC - **Use Case**: Pure vector similarity search
# MAGIC - **Configuration**: `QUERY_TYPE = "ANN"`
# MAGIC - **Result**: Finds vectors most similar to your query vector
# MAGIC 
# MAGIC #### **HYBRID (Text + Vector Search)**  
# MAGIC - **API Requirement**: Both `query_text` and `query_vector` allowed
# MAGIC - **Use Case**: Combines semantic text matching with vector similarity
# MAGIC - **Configuration**: `QUERY_TYPE = "HYBRID"`
# MAGIC - **Result**: Best of both text and vector matching
# MAGIC 
# MAGIC ### Ray Processing Benefits:
# MAGIC - **Distributed Processing**: Utilizes multiple worker nodes for parallel processing
# MAGIC - **Memory Efficient**: Processes data in batches without loading everything into memory
# MAGIC - **Scalable**: Automatically scales across your Spark cluster
# MAGIC - **Fault Tolerant**: Ray handles task failures and retries automatically
# MAGIC 
# MAGIC ### Performance Considerations:
# MAGIC - Adjust `batch_size` based on your cluster resources and memory constraints
# MAGIC - Increase `max_worker_nodes` for faster processing of large datasets
# MAGIC - Monitor resource usage and adjust CPU allocation as needed
# MAGIC 
# MAGIC ### Vector Format Notes:
# MAGIC - Vectors are automatically converted to list format for API compatibility
# MAGIC - Supports both numpy arrays and list formats as input
# MAGIC - Handles None/empty values for both text and vector queries
# MAGIC - Distributed processing maintains data integrity across worker nodes 