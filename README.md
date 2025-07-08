# Databricks Vector Search Examples

A comprehensive collection of examples for performing **batch vector search operations** using Databricks Vector Search with different processing approaches, query types, and configuration management.

## 🎯 Overview

This repository provides production-ready examples for scaling vector search operations across large datasets using:
- **Python Async Processing** (< 1M records, serverless CPU)
- **Ray Distributed Processing** (> 1M records, multi-node clusters)
- **Centralized Configuration Management** (environment-specific settings)
- **Multiple Query Types** (ANN vector-only, HYBRID text+vector)

## 🚀 Quick Start

### Prerequisites
- Databricks workspace with Vector Search enabled
- Unity Catalog configured
- Python libraries: `databricks-vectorsearch`, `httpx`, `ray[default]`

### 1. Data Preparation
```bash
# Download and prepare the IMDB dataset
Run: vector_search_batch/01-download-dataset.ipynb
```

### 2. Index Creation
```bash
# Create vector search index with embeddings
Run: vector_search_batch/02-create-vector-search-index.ipynb
```

### 3. Configuration Setup
```python
# Choose your configuration approach
from vector_search_batch.config import VectorSearchConfig, ConfigPresets

# Option A: Default configuration (quickest start)
config = VectorSearchConfig()

# Option B: Environment preset
config = ConfigPresets.development()  # or staging, production

# Option C: Custom configuration
config = load_config(
    uc_catalog="your_catalog",
    uc_schema="your_schema",
    default_query_type="ANN"
)
```

### 4. Run Batch Processing
```bash
# For datasets < 1M records
Run: vector_search_batch/03-vs-async-batch-python.py

# For datasets > 1M records
Run: vector_search_batch/03-vs-async-batch-ray.py
```

## 📁 Repository Structure

```
databricks-vector-search-examples/
├── README.md                          # This file
├── databricks.yml                     # Databricks configuration
└── vector_search_batch/               # Main examples directory
    ├── README.md                      # Detailed documentation
    ├── config.py                      # 🆕 Configuration management system
    ├── 01-download-dataset.ipynb      # Data preparation
    ├── 02-create-vector-search-index.ipynb # Index creation
    ├── 03-vs-async-batch-python.py    # Python async processing
    ├── 03-vs-async-batch-ray.py       # Ray distributed processing
    └── [legacy files]                 # Older implementations
```

## 🔧 Configuration System (NEW!)

### Key Features
- ✅ **Type-safe configuration** with dataclass validation
- ✅ **Multiple configuration sources** (default, presets, environment variables, overrides)
- ✅ **Environment-specific settings** (dev, staging, production)
- ✅ **Easy customization** without editing core code
- ✅ **CI/CD integration** via environment variables

### Configuration Options

#### Default Configuration
```python
from vector_search_batch.config import VectorSearchConfig
config = VectorSearchConfig()  # Uses sensible defaults
```

#### Environment Presets
```python
from vector_search_batch.config import ConfigPresets

config = ConfigPresets.development()  # Lower concurrency, smaller samples
config = ConfigPresets.staging()      # Medium settings
config = ConfigPresets.production()   # High performance settings
```

#### Environment Variables
```bash
export UC_CATALOG="prod"
export UC_SCHEMA="vector_search"
export VS_INDEX_NAME="prod_vs_index"
export DEFAULT_QUERY_TYPE="HYBRID"
export DEFAULT_CONCURRENCY="100"
```

```python
from vector_search_batch.config import load_config
config = load_config(use_env=True)
```

#### Custom Overrides
```python
from vector_search_batch.config import load_config
config = load_config(
    uc_catalog="my_catalog",
    default_query_type="ANN",
    default_concurrency=50,
    max_sample_size=1000
)
```

## 🎯 Processing Approaches

### Python Async Processing
**Best for**: < 1M records, serverless CPU, rapid prototyping

```python
# Features:
- Configurable concurrency (default: 100)
- Automatic retry logic with exponential backoff
- Memory-efficient processing
- All query types supported (ANN, HYBRID)
- Serverless CPU compatible
```

### Ray Distributed Processing
**Best for**: > 1M records, multi-node clusters, production scale

```python
# Features:
- Distributed processing across multiple worker nodes
- Memory-efficient batch processing
- Automatic fault tolerance and retries
- Scales to very large datasets
- Advanced resource management
```

## 🔍 Query Types

### ANN (Vector-Only Search)
- **Use Case**: Pure vector similarity search
- **API**: Only `query_vector` parameter
- **Best For**: Finding semantically similar content based on embeddings

### HYBRID (Text + Vector Search)
- **Use Case**: Combines semantic text matching with vector similarity
- **API**: Both `query_text` and `query_vector` parameters
- **Best For**: Balanced search combining text and vector matching

## 📊 Performance Comparison

| Aspect | Python Async | Ray Distributed |
|--------|-------------|----------------|
| **Dataset Size** | < 1M records | > 1M records |
| **Memory Usage** | Moderate | Very Low |
| **Setup Complexity** | Simple | Moderate |
| **Fault Tolerance** | Basic retry | Advanced |
| **Scalability** | Vertical | Horizontal |
| **Compute Type** | Single-node | Multi-node |

## 🎯 Use Case Examples

### Movie Recommendation System
```python
config = load_config(
    default_query_type="HYBRID",    # Text + vector similarity
    default_num_results=10,         # Top 10 recommendations
    default_concurrency=50
)
```

### Content Deduplication
```python
config = load_config(
    default_query_type="ANN",       # Pure vector similarity
    default_num_results=5,          # Find top duplicates
    default_concurrency=100
)
```

### Large-Scale Production Processing
```python
config = ConfigPresets.production()
# Use Ray distributed processing for > 1M records
```

## 🔧 Advanced Configuration

### Environment-Specific Deployments
```python
# Development
config = ConfigPresets.development()
# - Lower concurrency (20)
# - Smaller sample size (100)
# - Faster timeouts (15s)

# Production  
config = ConfigPresets.production()
# - High concurrency (100)
# - Large sample size (10000)
# - Longer timeouts (60s)
```

### Custom Configuration for Specific Use Cases
```python
# High-throughput processing
config = load_config(
    default_concurrency=200,
    max_sample_size=50000,
    request_timeout=60
)

# Memory-constrained environment
config = load_config(
    default_concurrency=20,
    max_sample_size=100,
    request_timeout=15
)
```

## 🛠️ Installation and Setup

### 1. Clone Repository
```bash
git clone https://github.com/databricks/databricks-vector-search-examples.git
cd databricks-vector-search-examples
```

### 2. Install Dependencies
```bash
%pip install databricks-vectorsearch httpx ray[default]
```

### 3. Configure Environment (Optional)
```bash
# Set environment variables for your deployment
export UC_CATALOG="your_catalog"
export UC_SCHEMA="your_schema"
export VS_INDEX_NAME="your_index"
export VECTOR_SEARCH_ENDPOINT="your_endpoint"
```

### 4. Run Examples
```bash
# Start with data preparation
# Then run either Python async or Ray distributed processing
```

## 🔍 Troubleshooting

### Configuration Issues
- **Config not found**: Ensure `config.py` is in the correct directory
- **Environment variables not loading**: Use `load_config(use_env=True)`
- **Validation errors**: Check parameter types and ranges

### Performance Issues
- **Memory errors**: Use Ray distributed approach or reduce `max_sample_size`
- **Rate limiting**: Reduce `default_concurrency` or use development preset
- **Timeouts**: Increase `request_timeout` or use production preset

## 📚 Documentation

- **Detailed Documentation**: See [vector_search_batch/README.md](vector_search_batch/README.md)
- **Configuration Reference**: See [vector_search_batch/config.py](vector_search_batch/config.py)
- **API Documentation**: [Databricks Vector Search](https://docs.databricks.com/machine-learning/vector-search.html)

## 🤝 Contributing

1. **Use the configuration system** for all new examples
2. **Test with all presets** (development, staging, production)
3. **Follow existing naming conventions**
4. **Include comprehensive documentation**
5. **Add performance benchmarks**

## 📄 License

This repository is provided as examples for Databricks customers and field engineering teams.

## 🆘 Support

For issues and questions:
- Check the troubleshooting section in [vector_search_batch/README.md](vector_search_batch/README.md)
- Review configuration documentation
- Open an issue in this repository
