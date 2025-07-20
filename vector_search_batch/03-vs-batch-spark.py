# Databricks notebook source
# MAGIC %md ## Spark Pandas UDF for Vector Search with Concurrency Control
# MAGIC - Distributes requests across Spark partitions
# MAGIC - Controls concurrency within each partition using threading
# MAGIC - Supports both HYBRID and ANN search types
# MAGIC - Handles retry logic and error handling
# MAGIC - Compatible with your existing vector search index

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch httpx
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Import required libraries
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType
import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any
import threading
from dataclasses import asdict

# Import configuration
from config import VectorSearchConfig, ConfigPresets, load_config

# COMMAND ----------

# Initialize configuration
config = load_config(default_concurrency=50, max_sample_size=5000, vector_search_endpoint="storage_optimized_test")
config.print_config()

# Set variables for backward compatibility
UC_CATALOG = config.uc_catalog
UC_SCHEMA = config.uc_schema
VS_INDEX_NAME = config.vs_index_name
VECTOR_SEARCH_ENDPOINT = config.vector_search_endpoint
VECTOR_SEARCH_INDEX = config.vector_search_index
EMBEDDING_DIMENSION = config.embedding_dimension
SOURCE_DATASET = config.source_dataset
ID_COLUMN = config.id_column
EMBEDDINGS_COLUMN = config.embeddings_column
TEXT_COLUMN = config.text_column

# COMMAND ----------

# Vector Search API functions (synchronous versions)
def get_workspace_config():
    """Get workspace URL and token."""
    WORKSPACE_URL = dbutils.entry_point.getDbutils().notebook().getContext().apiUrl().get()
    TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    return WORKSPACE_URL, TOKEN

def build_headers(token: str) -> Dict[str, str]:
    """Build HTTP headers for API requests."""
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

def build_payload(
    query_text: Optional[str] = None,
    query_vector: Optional[List[float]] = None,
    columns: Optional[List[str]] = None,
    num_results: int = 5,
    query_type: str = "HYBRID",
    filters_json: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Build payload for vector search API."""
    payload = {
        "num_results": num_results,
        "columns": columns or [ID_COLUMN, EMBEDDINGS_COLUMN, TEXT_COLUMN],
        "query_type": query_type
    }
    
    if query_type == "ANN":
        if query_vector is not None and len(query_vector) > 0:
            payload["query_vector"] = query_vector
    elif query_type == "HYBRID":
        if query_text is not None and query_text.strip():
            payload["query_text"] = query_text
        if query_vector is not None and len(query_vector) > 0:
            payload["query_vector"] = query_vector
    else:
        if query_text is not None and query_text.strip():
            payload["query_text"] = query_text
    
    if filters_json is not None:
        payload["filters_json"] = filters_json
    
    payload.update(kwargs)
    return payload

def query_vector_search_sync(
    workspace_url: str,
    token: str,
    index_name: str,
    payload: Dict[str, Any],
    max_retries: int = 3,
    backoff_factor: float = 2.0
) -> Dict[str, Any]:
    """Synchronous API call with retry logic."""
    headers = build_headers(token)
    url = f"{workspace_url}/api/2.0/vector-search/indexes/{index_name}/query"
    
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429 or response.status_code >= 500:
                wait_time = backoff_factor ** attempt
                print(f"Retrying in {wait_time} seconds due to status {response.status_code}...")
                time.sleep(wait_time)
            else:
                print(f"Failed: {response.status_code} - {response.text}")
                break
                
        except requests.RequestException as e:
            print(f"Request error: {e}, retrying...")
            if attempt < max_retries:
                time.sleep(backoff_factor ** attempt)
    
    return {}

def parse_response(data: Dict[str, Any], fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Parse API response to extract desired fields."""
    if not data or "result" not in data or "data_array" not in data["result"]:
        return []
    
    columns = [col['name'] for col in data['manifest']['columns']]
    col_idx = {name: idx for idx, name in enumerate(columns)}
    fields = fields or [ID_COLUMN, EMBEDDINGS_COLUMN, TEXT_COLUMN, "score"]
    
    rows = []
    for row in data['result']['data_array']:
        row_dict = {}
        for field in fields:
            if field in col_idx:
                row_dict[field] = row[col_idx[field]]
        rows.append(row_dict)
    
    return rows

# COMMAND ----------

# Define the Spark Pandas UDF
def create_vector_search_udf(
    concurrency: int = 50,
    num_results: int = 5,
    query_type: str = "HYBRID",
    columns: Optional[List[str]] = None
):
    """Create a Pandas UDF for vector search with concurrency control."""
    
    # Get workspace config (this will be serialized with the UDF)
    workspace_url, token = get_workspace_config()
    index_name = VECTOR_SEARCH_INDEX
    search_columns = columns or [ID_COLUMN, EMBEDDINGS_COLUMN, TEXT_COLUMN]
    
    def vector_search_worker(args):
        """Worker function for individual vector search requests."""
        idx, query_text, query_vector, lookup_id = args
        
        try:
            # Convert vector to list if needed
            if query_vector is not None:
                if hasattr(query_vector, 'tolist'):
                    query_vector = query_vector.tolist()
                elif isinstance(query_vector, np.ndarray):
                    query_vector = query_vector.tolist()
                elif not isinstance(query_vector, list):
                    query_vector = list(query_vector)
            
            # Build payload
            payload = build_payload(
                query_text=query_text,
                query_vector=query_vector,
                columns=search_columns,
                num_results=num_results,
                query_type=query_type
            )
            
            # Make API call
            response = query_vector_search_sync(
                workspace_url=workspace_url,
                token=token,
                index_name=index_name,
                payload=payload,
                max_retries=config.max_retries,
                backoff_factor=config.backoff_factor
            )
            
            # Parse response
            results = parse_response(response, fields=search_columns + ["score"])
            
            # Add lookup information to each result
            for result in results:
                result["lookup_id"] = lookup_id
                result["lookup_text"] = query_text
                result["query_index"] = idx
            
            return results
            
        except Exception as e:
            print(f"Error in vector search for index {idx}: {str(e)}")
            return []
    
    def vector_search_udf(
        query_texts: pd.Series,
        query_vectors: pd.Series,
        lookup_ids: pd.Series
    ) -> pd.Series:
        """
        Pandas UDF function for vector search.
        
        Args:
            query_texts: Series of text queries
            query_vectors: Series of embedding vectors
            lookup_ids: Series of lookup IDs
            
        Returns:
            Series of JSON strings containing search results
        """
        try:
            # Prepare arguments for workers
            args_list = []
            for idx, (query_text, query_vector, lookup_id) in enumerate(
                zip(query_texts, query_vectors, lookup_ids)
            ):
                args_list.append((idx, query_text, query_vector, lookup_id))
            
            # Execute with thread pool
            all_results = []
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                # Submit all tasks
                future_to_args = {
                    executor.submit(vector_search_worker, args): args 
                    for args in args_list
                }
                
                # Collect results
                for future in as_completed(future_to_args):
                    try:
                        results = future.result()
                        all_results.extend(results)
                    except Exception as e:
                        print(f"Worker task failed: {str(e)}")
            
            # Convert results to JSON strings for return
            result_jsons = []
            for i in range(len(query_texts)):
                # Filter results for this query index
                query_results = [
                    r for r in all_results 
                    if r.get("query_index") == i
                ]
                result_jsons.append(json.dumps(query_results))
            
            return pd.Series(result_jsons)
            
        except Exception as e:
            print(f"Error in vector_search_udf: {str(e)}")
            # Return empty results for all queries
            return pd.Series([json.dumps([]) for _ in range(len(query_texts))])
    
    return vector_search_udf

# COMMAND ----------

# Create the UDF
vector_search_udf = create_vector_search_udf(
    concurrency=config.default_concurrency,
    num_results=config.default_num_results,
    query_type=config.default_query_type,
    columns=config.get_column_list()
)

# Register as Pandas UDF
@pandas_udf(returnType=StringType(), functionType=PandasUDFType.SCALAR)
def vector_search_pandas_udf(
    query_texts: pd.Series,
    query_vectors: pd.Series,
    lookup_ids: pd.Series
) -> pd.Series:
    return vector_search_udf(query_texts, query_vectors, lookup_ids)

# COMMAND ----------

# Test the UDF with sample data
source_df = spark.table(SOURCE_DATASET)
print(f"Source dataset has {source_df.count()} rows")

# Take a sample for testing
sample_df = source_df.select(
    F.col(ID_COLUMN).alias("lookup_id"),
    F.col(TEXT_COLUMN).alias("query_text"),
    F.col(EMBEDDINGS_COLUMN).alias("query_vector")
).limit(100)  # Start with small sample

print("Sample data schema:")
sample_df.printSchema()

# COMMAND ----------

# Apply the UDF to get search results
results_df = sample_df.withColumn(
    "search_results_json",
    vector_search_pandas_udf(
        F.col("query_text"),
        F.col("query_vector"),
        F.col("lookup_id")
    )
)

print("Results with UDF applied:")
results_df.show(5, truncate=False)

# COMMAND ----------

# Parse JSON results into structured format
from pyspark.sql.functions import from_json, explode, col

# Define schema for search results
result_schema = ArrayType(
    StructType([
        StructField("id", StringType(), True),
        StructField("embeddings", ArrayType(DoubleType()), True),
        StructField("text", StringType(), True),
        StructField("score", DoubleType(), True),
        StructField("lookup_id", StringType(), True),
        StructField("lookup_text", StringType(), True),
        StructField("query_index", IntegerType(), True)
    ])
)

# Parse JSON and explode results
final_results_df = results_df.withColumn(
    "search_results",
    from_json(col("search_results_json"), result_schema)
).select(
    col("lookup_id").alias("original_id"),
    col("query_text").alias("original_text"),
    explode(col("search_results")).alias("result")
).select(
    col("original_id"),
    col("original_text"),
    col("result.id").alias("found_id"),
    col("result.text").alias("found_text"),
    col("result.score").alias("similarity_score"),
    col("result.lookup_id"),
    col("result.query_index")
)

print(f"Final results: {final_results_df.count()} rows")
display(final_results_df.limit(20))

# COMMAND ----------

# MAGIC %md ### Performance Monitoring and Scaling

# COMMAND ----------

# Function to process larger datasets with monitoring
def process_large_dataset(
    df,
    batch_size: int = 1000,
    concurrency: int = 50,
    num_partitions: int = None
):
    """Process large dataset with monitoring and optimal partitioning."""
    
    total_rows = df.count()
    print(f"Processing {total_rows} rows with batch_size={batch_size}, concurrency={concurrency}")
    
    # Repartition for optimal processing
    if num_partitions is None:
        num_partitions = min(total_rows // batch_size + 1, 100)
    
    print(f"Using {num_partitions} partitions")
    
    # Repartition the dataframe
    df_partitioned = df.repartition(num_partitions)
    
    # Create UDF with specified concurrency
    udf_func = create_vector_search_udf(
        concurrency=concurrency,
        num_results=config.default_num_results,
        query_type=config.default_query_type,
        columns=config.get_column_list()
    )
    
    @pandas_udf(returnType=StringType(), functionType=PandasUDFType.SCALAR)
    def batch_vector_search_udf(
        query_texts: pd.Series,
        query_vectors: pd.Series,
        lookup_ids: pd.Series
    ) -> pd.Series:
        return udf_func(query_texts, query_vectors, lookup_ids)
    
    # Process with UDF
    results_df = df_partitioned.withColumn(
        "search_results_json",
        batch_vector_search_udf(
            F.col("query_text"),
            F.col("query_vector"),
            F.col("lookup_id")
        )
    )
    
    return results_df

# COMMAND ----------

# Example: Process a larger sample (uncomment to run)
large_sample_df = source_df.select(
    F.col(ID_COLUMN).alias("lookup_id"),
    F.col(TEXT_COLUMN).alias("query_text"),
    F.col(EMBEDDINGS_COLUMN).alias("query_vector")
)

if config.max_sample_size:
    large_sample_df = large_sample_df.limit(config.max_sample_size)

large_results_df = process_large_dataset(
    large_sample_df,
    batch_size=100,
    concurrency=30,
    num_partitions=10
)

# Parse JSON results into structured format
from pyspark.sql.functions import from_json, explode, col

# Define schema for search results
result_schema = ArrayType(
    StructType([
        StructField("id", StringType(), True),
        StructField("embeddings", ArrayType(DoubleType()), True),
        StructField("text", StringType(), True),
        StructField("score", DoubleType(), True),
        StructField("lookup_id", StringType(), True),
        StructField("lookup_text", StringType(), True),
        StructField("query_index", IntegerType(), True)
    ])
)

# Parse JSON and explode results
final_results_df = large_results_df.withColumn(
    "search_results",
    from_json(col("search_results_json"), result_schema)
).select(
    col("lookup_id").alias("original_id"),
    col("query_text").alias("original_text"),
    explode(col("search_results")).alias("result")
).select(
    col("original_id"),
    col("original_text"),
    col("result.id").alias("found_id"),
    col("result.text").alias("found_text"),
    col("result.score").alias("similarity_score"),
    col("result.lookup_id"),
    col("result.query_index")
)

# COMMAND ----------

# MAGIC %md ### Save Results to Delta Table

# COMMAND ----------

# Optional: Save results to Delta table
final_results_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{UC_CATALOG}.{UC_SCHEMA}.vector_search_results")
print(f"Results saved to {UC_CATALOG}.{UC_SCHEMA}.vector_search_results")

# COMMAND ----------

# MAGIC %md ### Load Results from Delta Table

# COMMAND ----------
final_results_df = spark.table(f"{UC_CATALOG}.{UC_SCHEMA}.vector_search_results")
print(f"Final results count: {final_results_df.count()}")
display(final_results_df.limit(20))
# COMMAND ----------

# MAGIC %md ### Summary
# MAGIC 
# MAGIC This notebook demonstrates how to convert async vector search code to a Spark Pandas UDF:
# MAGIC 
# MAGIC #### Key Features:
# MAGIC 1. **Spark Distribution**: Work is distributed across Spark partitions
# MAGIC 2. **Concurrency Control**: ThreadPoolExecutor controls concurrency within each partition
# MAGIC 3. **Retry Logic**: Built-in retry mechanism for failed requests
# MAGIC 4. **Flexible Configuration**: Support for different query types (HYBRID, ANN)
# MAGIC 5. **Error Handling**: Graceful handling of API errors and malformed data
# MAGIC 6. **Scalability**: Can process large datasets by adjusting partitioning and concurrency
# MAGIC 
# MAGIC #### Usage Tips:
# MAGIC - **Concurrency**: Start with lower values (20-50) and increase based on your endpoint limits
# MAGIC - **Partitioning**: Use `repartition()` to optimize parallel processing
# MAGIC - **Batch Size**: Process data in batches to avoid memory issues
# MAGIC - **Monitoring**: Monitor Spark UI for task performance and failures
# MAGIC 
# MAGIC #### Performance Considerations:
# MAGIC - Each partition will make concurrent API calls up to the specified limit
# MAGIC - Total concurrent requests = num_partitions × concurrency_per_partition
# MAGIC - Adjust based on your vector search endpoint rate limits
# MAGIC - Consider using Databricks clusters with sufficient CPU cores for optimal performance
