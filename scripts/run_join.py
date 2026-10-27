"""Join cleaned KPI data with static cell configuration and change history."""

import argparse
from pathlib import Path

import pandas as pd

from ranai.data.join import add_change_history, join_static_config
from ranai.data.load import load_config, load_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    columns = config["columns"]
    technology = config.get("technology", "lte")
    interim = Path(config["paths"]["interim"])
    kpi = pd.read_parquet(interim / f"{technology}_clean.parquet")
    cell_config = load_directory(Path(config["paths"]["raw"]) / "config")
    cell_config = cell_config.rename(columns=columns.get("column_mapping", {}))
    joined = join_static_config(kpi, cell_config, columns["cell_id"])

    changes_dir = Path(config["paths"]["raw"]) / "change_log"
    if changes_dir.is_dir() and any(changes_dir.iterdir()):
        changes = load_directory(changes_dir).rename(columns=columns.get("column_mapping", {}))
        joined = add_change_history(
            joined,
            changes,
            entity_column=columns["cell_id"],
            timestamp_column=columns["timestamp"],
            change_timestamp_column=columns.get("change_timestamp", "change_timestamp"),
        )
    interim.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(interim / f"{technology}_joined.parquet", index=False)


if __name__ == "__main__":
    main()
