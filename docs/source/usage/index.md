# Usage Guides

Comprehensive guides for using dc-tools in your data challenges and research.

## Getting Started

Start here if you're new to dc-tools:

- **[Quick Start (5 min)](quickstart.md)** - Get up and running in minutes
- **[Installation](../package_docs/installation.md)** - Detailed setup instructions

## Core Concepts

Understand the fundamentals:

- **[Configuration Guide](config.md)** - YAML-based evaluation workflows
- **[Architecture](architecture.md)** - System design and principles

## Common Tasks

Quick references for specific tasks:

### Data Loading

```python
from dctools.data import EvaluationDataloader

loader = EvaluationDataloader()
data = loader.load_dataset(source="argo")  # or "cmems", "s3"
```

See [Quick Start - Loading Data](quickstart.md#2-loading-data)

### Processing Data

```python
from dctools.processing import interpolate_dataset

interpolated = interpolate_dataset(model_data, reference_grid)
```

See [Quick Start - Processing Data](quickstart.md#4-processing-data)

### Computing Metrics

```python
from dctools.metrics import MetricComputer

computer = MetricComputer()
metrics = computer.compute(prediction, reference, ["rmse", "mae"])
```

See [Quick Start - Computing Metrics](quickstart.md#5-computing-metrics)

### Distributed Evaluation

```python
from dask.distributed import Client

with Client(n_workers=4):
    evaluator.evaluate()  # Automatically distributed
```

See [Quick Start - Distributed Evaluation](quickstart.md#6-distributed-evaluation-with-dask)

## Configuration

Learn to configure evaluation workflows:

- **[Configuration Basics](config.md#configuration-structure)** - YAML structure overview
- **[Data Sources](config.md#sources-section)** - Configure data inputs
- **[Evaluation Settings](config.md#evaluation-section)** - Specify metrics and parameters
- **[Output Options](config.md#output-section)** - Define result formats
- **[Complete Example](config.md#complete-example)** - Full working configuration

### Run from Configuration

```bash
poetry run python -m dctools run config.yaml
```

## Understanding Architecture

Detailed information about system design:

- **[High-Level Overview](architecture.md#high-level-architecture)** - System components
- **[Design Principles](architecture.md#core-design-principles)** - Key concepts
- **[Data Flow](architecture.md#data-flow)** - How data flows through system
- **[Component Details](architecture.md#component-details)** - Module breakdown
- **[Extension Points](architecture.md#extension-points)** - How to extend

## Advanced Topics

### Distributed Computing

Configure Dask for cluster execution:

```yaml
evaluation:
  dask:
    scheduler: distributed
    n_workers: 8
    memory_per_worker: 4GB
```

See [Configuration Guide - Distributed Computing](config.md#distributed-computing)

### Performance Optimization

Tips for faster evaluation:

1. **Use Zarr format** for large datasets
2. **Optimize chunks** based on your cluster
3. **Use profiles** (distributed) instead of gridded data when possible
4. **Cache remote data** locally for repeated runs

See [Architecture - Performance](architecture.md#performance-characteristics)

### Custom Implementations

Extend dc-tools for your needs:

```python
from dctools.processing import BaseDCEvaluation

class MyEvaluation(BaseDCEvaluation):
    # Custom logic here
    pass
```

See [Architecture - Extension Points](architecture.md#extension-points)

## Troubleshooting

### Installation Issues

See [Installation - Troubleshooting](../package_docs/installation.md#troubleshooting)

### Configuration Issues

See [Configuration - Troubleshooting](config.md#troubleshooting)

### Performance Issues

See [Architecture - Troubleshooting](architecture.md#troubleshooting-architecture-issues)

## Workflow Examples

### Simple Evaluation

```bash
# 1. Create config.yaml (5 min)
# 2. Run
poetry run python -m dctools run config.yaml
# 3. Check results
cat results/metrics.json
```

Time: ~30 minutes for typical dataset

### Large-Scale Evaluation

```bash
# 1. Setup cluster
dask-distributed scheduler &
dask-worker scheduler_address:8786 &

# 2. Configure for cluster (in config.yaml)
# 3. Run
poetry run python -m dctools run config.yaml

# 4. Monitor progress
# Visit http://localhost:8787 for Dask dashboard
```

Time: hours to days depending on dataset size

### Custom Challenge Implementation

```python
# 1. Create challenge class
from dctools.processing import BaseDCEvaluation

class MyChallenge(BaseDCEvaluation):
    VARIABLES = ['MY_VAR']
    METRICS = ['custom_metric']

# 2. Set parameters
# 3. Run evaluation
challenge = MyChallenge(config)
challenge.run()
```

Time: 1-2 hours implementation

## Best Practices

1. **Version your configurations** - Commit config.yaml to git
2. **Use lazy loading** - Always use Dask for large datasets
3. **Monitor memory** - Watch Dask dashboard during evaluation
4. **Cache data** - Reuse downloaded data across runs
5. **Test locally** - Run on small data before cluster
6. **Document assumptions** - Note physical ranges, units, etc.

## Next Steps

- **[Full API Reference](../package_docs/api.md)** - Complete module documentation
- **[Data Challenges](../data_challenges/dc_index.md)** - Challenge-specific guides
- **[GitHub Issues](https://github.com/ppr-ocean-ia/dc-tools/issues)** - Report problems
- **[GitHub Discussions](https://github.com/ppr-ocean-ia/dc-tools/discussions)** - Ask questions

---

```{toctree}
:maxdepth: 2
:hidden:

quickstart.md
config.md
architecture.md
```