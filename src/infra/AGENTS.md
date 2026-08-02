# Infrastructure Agents

## OVERVIEW
Infrastructure management including deployment, error recovery, and system rollback.

## WHERE TO LOOK
| Task | File | Role |
| :--- | :--- | :--- |
| Deployment | `deployment_manager.py` | Manages versioning, environments, and artifact deployment |
| Rollback | `rollback_system.py` | Handles checkpoints and recovery plans |
| Error Recovery | `error_recovery.py` | Implements retry policies and timeout management |
| Migrations | `migration_system.py` | Manages database and schema migrations |
| Notifications | `notification_system.py` | Centralized system status and error alerting |
| Task Execution | `task_service.py` | Manages background infrastructure tasks |

## CONVENTIONS
- Use `DeploymentStatus`, `RollbackStatus`, and `MigrationStatus` enums for state tracking.
- All infrastructure operations must be thread-safe (use `RLock`).
- Errors must be reported via `SystemNotifier`.

## ANTI-PATTERNS
- Do not bypass `DeploymentManager` when deploying new versions.
- Avoid manual file manipulation; use `RollbackSystem` for state changes.
- Never use bare `except: pass` in recovery logic.
