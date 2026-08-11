from pathlib import Path
import csv
import json
import argparse


def batch_number(path: Path) -> int:
    return int(path.name.split("_")[1])


def parse_experiment_name(name: str):
    parts = name.split("_")

    cps = int(parts[1])
    epsilon = float(parts[3].replace("p", "."))

    return cps, epsilon


def combine_experiment(experiment_dir: Path):
    experiment_name = experiment_dir.name
    output_dir = experiment_dir.parent

    output_jsonl = output_dir / f"{experiment_name}_attack_results.jsonl"
    output_validated_jsonl = output_dir / f"{experiment_name}_validated_attack_results.jsonl"
    output_csv = output_dir / f"{experiment_name}_attack_history.csv"

    # Skip if already processed
    if output_jsonl.exists() and output_validated_jsonl.exists() and output_csv.exists():
        print(f"[SKIP] {experiment_name} already processed")
        return

    cps, epsilon = parse_experiment_name(experiment_name)

    batch_dirs = sorted(
        experiment_dir.glob("batch_*"),
        key=batch_number,
    )

    print(f"[PROCESS] {experiment_name}")
    print(f"  Found {len(batch_dirs)} batches")

    # Combine attack_results.jsonl
    total_results = 0

    with output_jsonl.open("w", encoding="utf-8") as outfile:
        for batch_dir in batch_dirs:
            input_file = batch_dir / "attack_results.jsonl"

            if not input_file.exists():
                print(f"  [WARNING] Missing {input_file}")
                continue

            with input_file.open("r", encoding="utf-8") as infile:
                for line in infile:
                    line = line.strip()

                    if not line:
                        continue

                    record = json.loads(line)

                    record["experiment"] = experiment_name
                    record["batch"] = batch_dir.name

                    outfile.write(json.dumps(record) + "\n")
                    total_results += 1

    # Combine validated_attack_results.jsonl
    total_validated_results = 0

    with output_validated_jsonl.open("w", encoding="utf-8") as outfile:
        for batch_dir in batch_dirs:
            input_file = batch_dir / "validated_attack_results.jsonl"

            if not input_file.exists():
                print(f"  [WARNING] Missing {input_file}")
                continue

            with input_file.open("r", encoding="utf-8") as infile:
                for line in infile:
                    line = line.strip()

                    if not line:
                        continue

                    record = json.loads(line)

                    record["experiment"] = experiment_name
                    record["batch"] = batch_dir.name

                    outfile.write(json.dumps(record) + "\n")
                    total_validated_results += 1

    # Combine attack_history.csv
    total_history_rows = 0
    writer = None

    with output_csv.open("w", newline="", encoding="utf-8") as outfile:
        for batch_dir in batch_dirs:
            input_file = batch_dir / "attack_history.csv"

            if not input_file.exists():
                print(f"  [WARNING] Missing {input_file}")
                continue

            with input_file.open("r", newline="", encoding="utf-8") as infile:
                reader = csv.DictReader(infile)

                if reader.fieldnames is None:
                    continue

                if writer is None:
                    extra_fields = [
                        "experiment",
                        "batch",
                    ]

                    fieldnames = list(reader.fieldnames)

                    for field in extra_fields:
                        if field not in fieldnames:
                            fieldnames.append(field)

                    writer = csv.DictWriter(
                        outfile,
                        fieldnames=fieldnames,
                    )
                    writer.writeheader()

                for row in reader:
                    row["experiment"] = experiment_name
                    row["control_point_spacing"] = cps
                    row["epsilon"] = epsilon
                    row["batch"] = batch_dir.name

                    writer.writerow(row)
                    total_history_rows += 1

    print(f"  Results: {total_results}")
    print(f"  Validated results: {total_validated_results}")
    print(f"  History rows: {total_history_rows}")
    print(f"  Saved: {output_jsonl.name}")
    print(f"  Saved: {output_validated_jsonl.name}")
    print(f"  Saved: {output_csv.name}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("result/MedVLM-R1/bias_field_attack"),
        help="Root bias_field_attack directory",
    )

    args = parser.parse_args()

    experiment_dirs = sorted(
        path
        for path in args.root.glob("cps_*")
        if path.is_dir()
    )

    print(f"Found {len(experiment_dirs)} configurations\n")

    for experiment_dir in experiment_dirs:
        combine_experiment(experiment_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()