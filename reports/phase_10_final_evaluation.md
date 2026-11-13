# Phase 10 final evaluation — partial access

## Status

The authorized one-pass evaluation of the current URL-only and URL+HTML models was interrupted before it could produce a complete result. The evaluator completed shard 1 of 21 (1,000 rows) and had started processing later shards; the total rows processed is unknown.

During a process-status check, a separate shell command also read `data/raw/data/test-000.parquet` without selecting columns and discarded its output. The evaluator was stopped when this extra read was noticed. No test metrics or aggregate evaluation artifacts were produced, and the pass was not resumed.

The incident and pre-access record are documented in `reports/test_set_access_log.md`. Any future official-test read requires fresh user authorization and a new pre-access log entry.
