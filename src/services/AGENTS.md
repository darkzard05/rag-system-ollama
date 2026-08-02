# SERVICES AGENTS

## OVERVIEW
High-performance service layer for monitoring, optimization, and distributed RAG operations.

## STRUCTURE
* `distributed/`: Cluster management and distributed embedding/search.
* `monitoring/`: System health, metrics aggregation, and anomaly detection.
* `optimization/`: Performance tuning for async, batch, cache, and vector DBs.

## WHERE TO LOOK
| Task | File | Role |
| :--- | :--- | :--- |
| Cluster Management | `src/services/distributed/cluster_manager.py` | Manages distributed nodes |
| Distributed Search | `src/services/distributed/distributed_search.py` | Orchestrates multi-node search |
| System Health | `src/services/monitoring/health_checker.py` | Monitors node/service status |
| Metrics Aggregation | `src/services/monitoring/metrics_aggregator.py` | Aggregates time-series metrics |
| Anomaly Detection | `src/services/monitoring/metrics_aggregator.py` | Detects spikes/dips in metrics |
| LLM Usage Tracking | `src/services/monitoring/llm_tracker.py` | Monitors LLM token usage and costs |
| Performance Monitoring | `src/services/monitoring/performance_monitor.py` | Tracks system performance |
| Async Optimization | `src/services/optimization/async_optimizer.py` | Optimizes concurrent task execution |
| Batch Processing | `src/services/optimization/batch_optimizer.py` | Optimizes batch embedding/retrieval |
| Cache Tuning | `src/services/optimization/caching_optimizer.py` | Manages cache hit rates and TTL |
| Vector DB Tuning | `src/services/optimization/vector_db_optimizer.py` | Optimizes index and search performance |
| RAG Evaluation | `src/services/evaluation_service.py` | Runs Ragas-based quality checks |

## CONVENTIONS
* Use `logging` for all service-level events.
* Mandatory type annotations for all public methods.
* Use `threading.RLock` for thread-safe state management within services.
* Prefer `asyncio` for I/O bound distributed tasks.

## ANTI-PATTERNS
* No direct UI calls from services.
* No hardcoded thresholds; use `config.yml` or service parameters.
* Avoid long-running synchronous blocks in `async` services.
