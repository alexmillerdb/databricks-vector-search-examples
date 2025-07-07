# Databricks notebook source
# MAGIC %md ## Async Hybrid Search with Configurable Concurrency using `asyncio` and `httpx`
# MAGIC - Loads the Spark dataframe, selects both `embeddings` (vectors) and `text` columns
# MAGIC - Converts vectors to Python list format and runs async processes with automatic retry logic
# MAGIC - Supports **HYBRID search** combining both text and vector similarity
# MAGIC - Good for datasets < 1M records
# MAGIC - Can use Serverless CPU compute
# MAGIC - **Compatible with IMDB embeddings dataset from 02-create-vector-search-index.ipynb**
# MAGIC 
# MAGIC ### Important Notes:
# MAGIC - This notebook assumes your dataset has columns: `id`, `embeddings`, `text`
# MAGIC - Uses **HYBRID search** combining both text and vector similarity
# MAGIC - Automatically converts vectors to list format for API compatibility
# MAGIC - Handles None/empty values for both text and vector queries
# MAGIC - Verify the schema using the "Data Schema Validation" section below

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch httpx
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

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
TEXT_COLUMN = "text"    # Or None if you don't want to include the text column

source_df = spark.table(SOURCE_DATASET)

# COMMAND ----------

source_df.count()

# COMMAND ----------

# MAGIC %md ### Data Schema Validation

# COMMAND ----------

# Display the schema to verify column names match the configuration
print("Dataset Schema:")
source_df.printSchema()

# Display sample data
print("\nSample Data:")
display(source_df.limit(5))

# COMMAND ----------

# MAGIC %md ### Single REST API Call Example (Vector Search)

# COMMAND ----------

# single example text query
single_row_example = source_df.select(EMBEDDINGS_COLUMN, ID_COLUMN, TEXT_COLUMN).limit(1).toPandas()
query_vector = single_row_example[EMBEDDINGS_COLUMN][0]
record_id = single_row_example[ID_COLUMN][0]
query_text = single_row_example[TEXT_COLUMN][0]

print("Query Vector: ", query_vector) # This is the vector to search for
print("Record ID: ", record_id)
print("Query Text: ", query_text)

# COMMAND ----------

import requests
import json

# Configuration
WORKSPACE_URL = dbutils.entry_point.getDbutils().notebook().getContext().apiUrl().get()
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
INDEX_NAME = VECTOR_SEARCH_INDEX
columns_to_include = [ID_COLUMN, EMBEDDINGS_COLUMN, TEXT_COLUMN]

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
    "query_vector": query_vector.tolist(),
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
    "id": [],
    "embeddings": [],
    "text": [],
    "score": []
}

# Extract rows and populate the dictionary
for row in data['result']['data_array']:
    result_dict['id'].append(row[col_idx['id']])
    result_dict['embeddings'].append(row[col_idx['embeddings']])
    result_dict['text'].append(row[col_idx['text']])
    result_dict['score'].append(row[col_idx['score']])

# result_dict now has the desired structure
print(result_dict)

# COMMAND ----------

# MAGIC %md ### Async Functions

# COMMAND ----------

import json
import asyncio
import httpx
from typing import List, Optional

def get_config(index_name=None):
    """Retrieve Databricks workspace URL, API token, and index name."""
    WORKSPACE_URL = dbutils.entry_point.getDbutils().notebook().getContext().apiUrl().get()
    TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    INDEX_NAME = index_name or VECTOR_SEARCH_INDEX
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
    fields = fields or [ID_COLUMN, EMBEDDINGS_COLUMN, TEXT_COLUMN, "score"]
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
    columns = columns or [ID_COLUMN, EMBEDDINGS_COLUMN, TEXT_COLUMN]

    semaphore = asyncio.Semaphore(concurrency)

    async def sem_task(query, query_vector, idx):
        async with semaphore:
            # Prepare query_text and query_vector based on query_type
            query_text_val = None
            query_vector_val = None
            
            # Handle different query types
            if query_type == "ANN":
                # ANN (vector-only): Only use query_vector, set query_text to None
                if query_vector is not None:
                    if hasattr(query_vector, 'tolist'):
                        query_vector_val = query_vector.tolist()
                    elif isinstance(query_vector, list):
                        query_vector_val = query_vector
                    else:
                        query_vector_val = query_vector
                # Explicitly set query_text to None for ANN
                query_text_val = None
                
            elif query_type == "HYBRID":
                # HYBRID: Use both query_text and query_vector
                if isinstance(query, str):
                    query_text_val = query
                
                if query_vector is not None:
                    if hasattr(query_vector, 'tolist'):
                        query_vector_val = query_vector.tolist()
                    elif isinstance(query_vector, list):
                        query_vector_val = query_vector
                    else:
                        query_vector_val = query_vector
                        
            else:
                # For other query types (like text-only), only use query_text
                if isinstance(query, str):
                    query_text_val = query
                query_vector_val = None
            
            payload = build_payload(
                query_text=query_text_val,
                query_vector=query_vector_val,
                columns=columns,
                num_results=num_results,
                query_type=query_type,
                filters_json=filters_json,
                **kwargs
            )
            # Debug: Print payload for first few requests
            if idx < 3:
                print(f"Request {idx} payload keys: {list(payload.keys())}")
                if "query_vector" in payload:
                    print(f"Request {idx} vector dimensions: {len(payload['query_vector'])}")
                if "query_text" in payload:
                    print(f"Request {idx} query_text length: {len(payload['query_text'])}")
                print(f"Request {idx} columns: {payload.get('columns', [])}")
            
            async with httpx.AsyncClient(timeout=30) as client:
                response = await query_vector_search(client, workspace_url, index_name, headers, payload)
            parsed_results = parse_response(response, fields=columns + ["score"])
            # Attach lookup_content and lookup_id if provided
            for row in parsed_results:
                # Store the query used for lookup (could be text or vector representation)
                if query_text_val is not None:
                    row["lookup_content"] = query_text_val
                elif query_vector_val is not None:
                    row["lookup_embeddings"] = query_vector_val
                else:
                    row["lookup_content"] = str(query)
                
                if lookup_ids is not None and idx < len(lookup_ids):
                    row["lookup_id"] = lookup_ids[idx]
            return parsed_results

    # Prepare tasks with proper vector formatting
    tasks = []
    for i, query in enumerate(queries):
        query_vector = None
        if query_vector_list and i < len(query_vector_list):
            vec = query_vector_list[i]
            if vec is not None:
                # Ensure vector is properly formatted as list
                if hasattr(vec, 'tolist'):
                    query_vector = vec.tolist()
                elif isinstance(vec, list):
                    query_vector = vec
                else:
                    query_vector = list(vec)
        tasks.append(sem_task(query, query_vector, i))
    results = await asyncio.gather(*tasks)
    all_rows = [row for batch in results for row in batch]
    return all_rows

# COMMAND ----------

# MAGIC %md ### Run Async Hybrid Search with Configurable Concurrency
# MAGIC 
# MAGIC This section performs hybrid search using both text and vector queries:
# MAGIC - **Text queries**: From the `text` column for semantic text matching
# MAGIC - **Vector queries**: From the `embeddings` column for vector similarity
# MAGIC - **Query type**: HYBRID combines both approaches for better results
# MAGIC 
# MAGIC The function automatically handles:
# MAGIC - Vector format conversion (ensures vectors are lists)
# MAGIC - None/empty value checking for both text and vectors
# MAGIC - Proper payload construction based on available data

# COMMAND ----------

# Extract query vectors and text from the DataFrame
query_data_df = source_df.select(EMBEDDINGS_COLUMN, TEXT_COLUMN, ID_COLUMN).limit(1000).toPandas()

# Prepare vectors (convert to list format)
query_vectors = []
for vec in query_data_df[EMBEDDINGS_COLUMN]:
    if vec is not None:
        query_vectors.append(vec.tolist() if hasattr(vec, 'tolist') else list(vec))
    else:
        query_vectors.append(None)

# Prepare text queries
query_texts = query_data_df[TEXT_COLUMN].tolist()

# Prepare lookup IDs
lookup_ids = query_data_df[ID_COLUMN].tolist()

print(f"Prepared {len(query_vectors)} vector queries and {len(query_texts)} text queries for vector search.")

# COMMAND ----------

# MAGIC %md ### Data Validation Before Search

# COMMAND ----------

# Validate data before running search
print("=== Data Validation ===")
print(f"Index name: {VECTOR_SEARCH_INDEX}")
print(f"Columns to include: {[ID_COLUMN, EMBEDDINGS_COLUMN, TEXT_COLUMN]}")

# Check first few vectors
for i in range(min(3, len(query_vectors))):
    vec = query_vectors[i]
    text = query_texts[i] if i < len(query_texts) else "N/A"
    if vec is not None:
        vec_list = vec.tolist() if hasattr(vec, 'tolist') else vec
        print(f"Sample {i}: Vector dim={len(vec_list)}, Text length={len(str(text))}")
    else:
        print(f"Sample {i}: Vector is None, Text length={len(str(text))}")

# Validate that the index exists and columns are correct
print(f"\nChecking if columns exist in source table...")
try:
    sample_data = source_df.select(*[ID_COLUMN, EMBEDDINGS_COLUMN, TEXT_COLUMN]).limit(1).collect()
    print("✓ All columns exist in source table")
except Exception as e:
    print(f"✗ Column validation failed: {e}")

# COMMAND ----------

# MAGIC %md ### Run Async Hybrid Search with Controlled Concurrency
# COMMAND ----------

async def run_async_batch():
    return await async_vector_search_batch(
        queries=query_texts,  # Text queries (used for HYBRID, ignored for ANN)
        query_vector_list=query_vectors,  # Vectors (used for both HYBRID and ANN)
        lookup_ids=lookup_ids,
        index_name=VECTOR_SEARCH_INDEX,
        columns=[ID_COLUMN, EMBEDDINGS_COLUMN, TEXT_COLUMN],
        num_results=5,
        query_type="ANN",  # ANN=vector-only, HYBRID=text+vector, other=text-only
        concurrency=100
    )

# Execute the async function
# Note: In Databricks notebooks, you can use await directly (top-level await is supported)
all_rows = await run_async_batch()
# Create and display the Spark DataFrame
sdf = spark.createDataFrame(all_rows)
print(f"Spark DataFrame created with {sdf.count()} rows.")
display(sdf)

# COMMAND ----------

# MAGIC %md ### Optional: Save Results to Delta Table

# COMMAND ----------

# Uncomment to save results to a Delta table
# sdf.write.mode("overwrite").saveAsTable("users.alex_miller.imdb_vs_python_async_results")
# print("Results saved to Delta table: users.alex_miller.imdb_vs_python_async_results")

# COMMAND ----------

# MAGIC %md ### Configuration Summary
# MAGIC 
# MAGIC This notebook has been configured to work with:
# MAGIC - **Vector Search Index**: `users.alex_miller.vs_batch_example`
# MAGIC - **Source Dataset**: `users.alex_miller.imdb_embeddings`
# MAGIC - **Endpoint**: `abs_test_temp`
# MAGIC - **Expected Columns**: `id`, `embeddings`, `text`
# MAGIC - **Search Type**: Vector-based search (ANN) using embeddings
# MAGIC 
# MAGIC If your dataset has different column names, update the column references in:
# MAGIC 1. The configuration section (`ID_COLUMN`, `EMBEDDINGS_COLUMN`)
# MAGIC 2. The DataFrame selection operations (`.select(EMBEDDINGS_COLUMN, ID_COLUMN)`)
# MAGIC 3. The async function calls (`columns=[ID_COLUMN, EMBEDDINGS_COLUMN, TEXT_COLUMN]`)
# MAGIC 
# MAGIC ### Vector Format Notes:
# MAGIC - Vectors are automatically converted to list format for API compatibility
# MAGIC - Supports both numpy arrays and list formats as input
# MAGIC - Handles None/empty values for both text and vector queries
# MAGIC 
# MAGIC ### Query Type Options & API Behavior:
# MAGIC 
# MAGIC #### **ANN (Vector-Only Search)**
# MAGIC - **API Requirement**: Only `query_vector` allowed, `query_text` must be None
# MAGIC - **Use Case**: Pure vector similarity search
# MAGIC - **Configuration**: `query_type="ANN"`
# MAGIC - **Result**: Finds vectors most similar to your query vector
# MAGIC 
# MAGIC #### **HYBRID (Text + Vector Search)**  
# MAGIC - **API Requirement**: Both `query_text` and `query_vector` allowed
# MAGIC - **Use Case**: Combines semantic text matching with vector similarity
# MAGIC - **Configuration**: `query_type="HYBRID"`
# MAGIC - **Result**: Best of both text and vector matching
# MAGIC 
# MAGIC #### **Text-Only Search**
# MAGIC - **API Requirement**: Only `query_text` allowed, `query_vector` must be None
# MAGIC - **Use Case**: Traditional text-based search
# MAGIC - **Configuration**: Any query_type except "ANN", set `query_vector_list=None`
# MAGIC 
# MAGIC ### Current Configuration:
# MAGIC - **Mode**: ANN (Vector-only)
# MAGIC - **Behavior**: Uses only embeddings for search, ignores text queries
# MAGIC - **API Calls**: Include only `query_vector`, `query_text` is explicitly set to None 