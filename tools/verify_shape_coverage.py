#!/usr/bin/env python3
"""Check shape uniqueness and exact coverage of a pinned source manifest.
This verifies the declared source set, NOT global model/workload completeness.
Only the Python standard library is required. Never executes source model code.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

KEYS = ["gemm_batch_count", "M", "N", "K", "grouped_gemm_groups"]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def shape_key(row):
    vals = tuple(row[name].strip() for name in KEYS)
    if not all(v.isdigit() and int(v) > 0 for v in vals):
        raise ValueError(f"Expected positive integer dimensions: {vals}")
    return tuple(map(int, vals))


def verify(csv_path, source_path, report_path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if sha256(source_path) != report["provenance_sha256"]:
        raise ValueError("Provenance checksum mismatch")

    shapes = set()
    count = 0
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = shape_key(row)
            if k in shapes:
                raise ValueError(f"Duplicate shape: {k}")
            if not row.get("source_url") or not row.get("evidence_types"):
                raise ValueError(f"Missing source/evidence metadata: {k}")
            shapes.add(k)
            count += 1

    required = set()
    source_rows = 0
    with gzip.open(source_path, "rt", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            required.add(shape_key(row))
            if not all(row.get(k) for k in ["source_url", "source_commit", "evidence_type"]):
                raise ValueError("Incomplete source provenance")
            source_rows += 1

    if source_rows != report["audited_source_records"]:
        raise ValueError("Source record count mismatch")
    if len(required) != report["audited_unique_shapes"]:
        raise ValueError("Source unique-shape count mismatch")
    missing = required - shapes
    if missing:
        raise ValueError(f"{len(missing)} declared shapes missing; examples: {sorted(missing)[:10]}")

    exact_snapshot = sha256(csv_path) == report["output_sha256"]
    if exact_snapshot and count != report["output_unique_shapes"]:
        raise ValueError("Snapshot row count mismatch")

    return {
        "csv_rows": count,
        "unique_shapes": len(shapes),
        "audited_source_records": source_rows,
        "audited_unique_shapes": len(required),
        "missing_audited_shapes": len(missing),
        "matches_recorded_snapshot": exact_snapshot,
        "global_completeness_guaranteed": False,
        "scope": "Exact imported-source coverage only; not every model, workload or framework",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("concrete_shapes_deduplicated.csv"))
    parser.add_argument(
        "--sources", type=Path,
        default=Path("data/shape_audit/2026-10-26/source_shapes.csv.gz"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("data/shape_audit/2026-10-26/coverage.json"),
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.csv, args.sources, args.report), indent=2))


if __name__ == "__main__":
    main()
