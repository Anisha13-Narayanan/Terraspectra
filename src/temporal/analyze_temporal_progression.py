from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd()
INDEX_PATH = PROJECT_ROOT / "data" / "temporal_index.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "temporal_progression_summary.csv"


def main():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Temporal index not found: {INDEX_PATH}")

    df = pd.read_csv(INDEX_PATH)

    required = {"class", "day", "replicate", "filename", "mat_variable"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["day"] = pd.to_numeric(df["day"], errors="raise")

    summary = (
        df.groupby(["day", "class"])
        .size()
        .rename("observations")
        .reset_index()
    )

    totals = (
        df.groupby("day")
        .size()
        .rename("total_observations")
        .reset_index()
    )

    summary = summary.merge(totals, on="day", how="left")

    summary["percentage_of_day"] = (
        summary["observations"]
        / summary["total_observations"]
        * 100
    )

    summary = summary.sort_values(["day", "class"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)

    print("=" * 70)
    print("TERRASPECTRA TEMPORAL PROGRESSION ANALYSIS")
    print("=" * 70)
    print(f"Input:  {INDEX_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print()

    print("Observations by day:")
    print(
        df.groupby("day")
        .size()
        .rename("observations")
        .to_string()
    )

    print()
    print("Observations by day and class:")

    table = (
        summary
        .pivot(index="class", columns="day", values="observations")
        .fillna(0)
        .astype(int)
    )

    print(table.to_string())

    print()
    print("Percentage composition by day:")

    percentage = (
        summary
        .pivot(index="class", columns="day", values="percentage_of_day")
        .fillna(0)
        .round(2)
    )

    print(percentage.to_string())

    print()
    print("Detailed temporal summary:")
    print(summary.to_string(index=False))

    print()
    print(
        "NOTE: This summarizes labeled observations across days. "
        "It does not prove that the same individual plant was observed "
        "repeatedly across all four days."
    )

    print()
    print("Temporal progression analysis completed successfully.")


if __name__ == "__main__":
    main()