# Legacy PhreshPhish benchmark results are invalid

The legacy files `logs/phresh_benchmark_results.csv` and `logs/phresh_benchmark_summary.csv` are **INVALID** and must not be used as evidence of model quality.

The benchmark `.txt` files contain 64-character SHA-256 identifiers that refer to the dataset's `sha256` column. The legacy `src/evaluation/evaluate_benchmarks.py` passed those strings directly to the URL vectorizers and engineered URL feature extractor instead of joining them to dataset rows. Consequently, it scored hash text as if it were a URL and did not evaluate the benchmark examples represented by those hashes.

Phase 7 replaces the result with a test-only SHA-256 join in `src/evaluation/evaluate_phase7_final.py`. The new report records matched, unmatched, duplicate, and label-mismatch counts; per-file AP and P@R90; and aggregates by configured base rate. The old CSVs are retained for audit history but are not valid benchmark scores.