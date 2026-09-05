from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path.cwd()

INPUT_PATH = PROJECT_ROOT / "data" / "temporal_predictions.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "temporal_prediction_analysis.csv"


TRUE_CLASS_MAP = {
    "alternaria_alternata": "Alternaria alternata",
    "alternaria_solani": "Alternaria solani",
    "botrytis_cinerea": "Botrytis cinerea",
    "fusarium_oxysporum": "Fusarium oxysporum",
    "healthy": "Healthy",
}


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required = {
        "class",
        "day",
        "replicate",
        "predicted_class",
        "confidence",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["expected_class"] = df["class"].map(TRUE_CLASS_MAP)

    if df["expected_class"].isna().any():
        unknown = df.loc[
            df["expected_class"].isna(),
            "class"
        ].unique()

        raise ValueError(
            f"Unknown true classes: {unknown}"
        )

    df["correct"] = (
        df["predicted_class"]
        == df["expected_class"]
    )

    day_summary = (
        df.groupby("day")
        .agg(
            total_predictions=("correct", "size"),
            correct_predictions=("correct", "sum"),
            average_confidence=("confidence", "mean"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
    )

    class_day_summary = (
        df.groupby(["day", "class"])
        .agg(
            total_predictions=("correct", "size"),
            correct_predictions=("correct", "sum"),
            average_confidence=("confidence", "mean"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
    )

    class_summary = (
        df.groupby("class")
        .agg(
            total_predictions=("correct", "size"),
            correct_predictions=("correct", "sum"),
            average_confidence=("confidence", "mean"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    class_day_summary.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("=" * 70)
    print("TERRASPECTRA TEMPORAL PREDICTION ANALYSIS")
    print("=" * 70)

    print()
    print("Overall performance:")

    total = len(df)
    correct = int(df["correct"].sum())

    print(f"Total predictions: {total}")
    print(f"Correct predictions: {correct}")
    print(
        f"Overall accuracy: "
        f"{correct / total * 100:.2f}%"
    )

    print()
    print("Accuracy by day:")
    print(
        day_summary[
            [
                "day",
                "total_predictions",
                "correct_predictions",
                "average_confidence",
                "accuracy",
            ]
        ]
        .assign(
            accuracy=lambda x: x["accuracy"] * 100,
            average_confidence=lambda x:
                x["average_confidence"] * 100,
        )
        .round(2)
        .to_string(index=False)
    )

    print()
    print("Accuracy by class:")
    print(
        class_summary[
            [
                "class",
                "total_predictions",
                "correct_predictions",
                "average_confidence",
                "accuracy",
            ]
        ]
        .assign(
            accuracy=lambda x: x["accuracy"] * 100,
            average_confidence=lambda x:
                x["average_confidence"] * 100,
        )
        .round(2)
        .to_string(index=False)
    )

    print()
    print("Accuracy by day and class:")
    print(
        class_day_summary[
            [
                "day",
                "class",
                "total_predictions",
                "correct_predictions",
                "average_confidence",
                "accuracy",
            ]
        ]
        .assign(
            accuracy=lambda x: x["accuracy"] * 100,
            average_confidence=lambda x:
                x["average_confidence"] * 100,
        )
        .round(2)
        .to_string(index=False)
    )

    print()
    print("Most common predictions:")

    print(
        df.groupby("predicted_class")
        .size()
        .sort_values(ascending=False)
        .to_string()
    )

    print()
    print(f"Saved analysis: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()