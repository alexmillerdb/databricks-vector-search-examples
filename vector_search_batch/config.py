import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class VectorSearchConfig:
    """Configuration for vector search operations using dataclasses"""
    
    # UC Configuration
    uc_catalog: str = "users"
    uc_schema: str = "alex_miller"
    vs_index_name: str = "vs_batch_example"
    
    # Vector Search Configuration
    vector_search_endpoint: str = "abs_test_temp"
    embedding_dimension: int = 1024
    
    # Dataset Configuration
    source_table_name: str = "imdb_embeddings"
    id_column: str = "id"
    embeddings_column: str = "embeddings"
    text_column: str = "text"
    
    # Search Parameters
    default_num_results: int = 5
    default_query_type: str = "ANN"
    default_concurrency: int = 100
    max_sample_size: int = 1000
    
    # Retry Configuration
    max_retries: int = 5
    backoff_factor: float = 2.0
    request_timeout: int = 30
    
    # Computed properties (generated after init)
    vector_search_index: str = field(init=False)
    source_dataset: str = field(init=False)
    
    def __post_init__(self):
        """Compute derived properties after initialization"""
        self.vector_search_index = f"{self.uc_catalog}.{self.uc_schema}.{self.vs_index_name}"
        self.source_dataset = f"{self.uc_catalog}.{self.uc_schema}.{self.source_table_name}"
        self.validate()
    
    def validate(self):
        """Validate configuration values"""
        assert self.embedding_dimension > 0, "Embedding dimension must be positive"
        assert self.default_num_results > 0, "Number of results must be positive"
        assert self.default_concurrency > 0, "Concurrency must be positive"
        assert self.default_query_type in ["HYBRID", "ANN"], "Query type must be HYBRID or ANN"
        assert self.max_retries >= 0, "Max retries must be non-negative"
        assert self.backoff_factor > 0, "Backoff factor must be positive"
        assert self.request_timeout > 0, "Request timeout must be positive"
        print("✓ Configuration validated successfully")
    
    def get_column_list(self) -> List[str]:
        """Get list of columns for vector search"""
        return [self.id_column, self.embeddings_column, self.text_column]
    
    @classmethod
    def from_env(cls) -> 'VectorSearchConfig':
        """Load configuration from environment variables"""
        return cls(
            uc_catalog=os.getenv("UC_CATALOG", "users"),
            uc_schema=os.getenv("UC_SCHEMA", "alex_miller"),
            vs_index_name=os.getenv("VS_INDEX_NAME", "vs_batch_example"),
            vector_search_endpoint=os.getenv("VECTOR_SEARCH_ENDPOINT", "abs_test_temp"),
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "1024")),
            source_table_name=os.getenv("SOURCE_TABLE_NAME", "imdb_embeddings"),
            id_column=os.getenv("ID_COLUMN", "id"),
            embeddings_column=os.getenv("EMBEDDINGS_COLUMN", "embeddings"),
            text_column=os.getenv("TEXT_COLUMN", "text"),
            default_num_results=int(os.getenv("DEFAULT_NUM_RESULTS", "5")),
            default_query_type=os.getenv("DEFAULT_QUERY_TYPE", "HYBRID"),
            default_concurrency=int(os.getenv("DEFAULT_CONCURRENCY", "100")),
            max_sample_size=int(os.getenv("MAX_SAMPLE_SIZE", "1000")),
            max_retries=int(os.getenv("MAX_RETRIES", "5")),
            backoff_factor=float(os.getenv("BACKOFF_FACTOR", "2.0")),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "30"))
        )
    
    def print_config(self):
        """Print current configuration"""
        print("=== Vector Search Configuration ===")
        print(f"UC Catalog: {self.uc_catalog}")
        print(f"UC Schema: {self.uc_schema}")
        print(f"Index Name: {self.vs_index_name}")
        print(f"Full Index Path: {self.vector_search_index}")
        print(f"Source Dataset: {self.source_dataset}")
        print(f"Vector Search Endpoint: {self.vector_search_endpoint}")
        print(f"Embedding Dimension: {self.embedding_dimension}")
        print(f"Columns: {self.get_column_list()}")
        print(f"Default Query Type: {self.default_query_type}")
        print(f"Default Results: {self.default_num_results}")
        print(f"Default Concurrency: {self.default_concurrency}")
        print(f"Max Sample Size: {self.max_sample_size}")


class ConfigPresets:
    """Factory for creating configuration presets"""
    
    @staticmethod
    def development() -> VectorSearchConfig:
        """Development environment configuration"""
        return VectorSearchConfig(
            uc_catalog="dev",
            uc_schema="vector_search",
            vs_index_name="dev_vs_index",
            vector_search_endpoint="dev_vs_endpoint",
            source_table_name="dev_embeddings",
            max_sample_size=100,
            default_concurrency=20,
            max_retries=3,
            request_timeout=15
        )
    
    @staticmethod
    def staging() -> VectorSearchConfig:
        """Staging environment configuration"""
        return VectorSearchConfig(
            uc_catalog="staging",
            uc_schema="vector_search",
            vs_index_name="staging_vs_index",
            vector_search_endpoint="staging_vs_endpoint",
            source_table_name="staging_embeddings",
            max_sample_size=1000,
            default_concurrency=50,
            max_retries=5,
            request_timeout=30
        )
    
    @staticmethod
    def production() -> VectorSearchConfig:
        """Production environment configuration"""
        return VectorSearchConfig(
            uc_catalog="prod",
            uc_schema="vector_search",
            vs_index_name="prod_vs_index",
            vector_search_endpoint="prod_vs_endpoint",
            source_table_name="prod_embeddings",
            max_sample_size=10000,
            default_concurrency=100,
            max_retries=5,
            request_timeout=60
        )
    
    @staticmethod
    def testing() -> VectorSearchConfig:
        """Testing environment configuration"""
        return VectorSearchConfig(
            uc_catalog="test",
            uc_schema="vector_search",
            vs_index_name="test_vs_index",
            vector_search_endpoint="test_vs_endpoint",
            source_table_name="test_embeddings",
            max_sample_size=10,
            default_concurrency=5,
            max_retries=2,
            request_timeout=10
        )
    
    @staticmethod
    def custom(
        catalog: str,
        schema: str,
        index_name: str,
        endpoint: str,
        table_name: str,
        **kwargs
    ) -> VectorSearchConfig:
        """Create custom configuration"""
        return VectorSearchConfig(
            uc_catalog=catalog,
            uc_schema=schema,
            vs_index_name=index_name,
            vector_search_endpoint=endpoint,
            source_table_name=table_name,
            **kwargs
        )


def load_config(preset: Optional[str] = None, use_env: bool = False, **overrides) -> VectorSearchConfig:
    """Load configuration with multiple sources"""
    
    # Start with default config
    config = VectorSearchConfig()
    
    # Load from preset if specified
    if preset:
        preset_configs = {
            "development": ConfigPresets.development,
            "staging": ConfigPresets.staging,
            "production": ConfigPresets.production,
            "testing": ConfigPresets.testing
        }
        
        if preset in preset_configs:
            config = preset_configs[preset]()
            print(f"✓ Loaded preset: {preset}")
        else:
            print(f"⚠ Unknown preset: {preset}. Using default.")
    
    # Load from environment variables if enabled
    if use_env:
        config = VectorSearchConfig.from_env()
        print("✓ Loaded configuration from environment variables")
    
    # Apply direct overrides
    if overrides:
        config_dict = {
            'uc_catalog': config.uc_catalog,
            'uc_schema': config.uc_schema,
            'vs_index_name': config.vs_index_name,
            'vector_search_endpoint': config.vector_search_endpoint,
            'embedding_dimension': config.embedding_dimension,
            'source_table_name': config.source_table_name,
            'id_column': config.id_column,
            'embeddings_column': config.embeddings_column,
            'text_column': config.text_column,
            'default_num_results': config.default_num_results,
            'default_query_type': config.default_query_type,
            'default_concurrency': config.default_concurrency,
            'max_sample_size': config.max_sample_size,
            'max_retries': config.max_retries,
            'backoff_factor': config.backoff_factor,
            'request_timeout': config.request_timeout
        }
        config = VectorSearchConfig(**{**config_dict, **overrides})
        print(f"✓ Applied overrides: {list(overrides.keys())}")
    
    return config 