import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def add_labels(ax, fmt="%.4f"):
    """Helper to add labels to the top of bars and adjust Y-axis height."""
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, padding=3, fontsize=10)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.1)


def save_dataframe_exports(df, output_dir, base_name):
    """Save dataframe as parquet, csv, and markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f"{base_name}.parquet"
    csv_path = output_dir / f"{base_name}.csv"
    md_path = output_dir / f"{base_name}.md"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))

    print(f"Saved averaged dataframe to:")
    print(f"  - {parquet_path}")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")


def canonicalize_experiment_names(df, model_students_dir=None):
    """Use saved student-directory names as canonical experiment labels."""
    if model_students_dir is None or "EXPERIMENT" not in df.columns:
        return df

    model_students_dir = Path(model_students_dir)
    if not model_students_dir.is_dir():
        return df

    canonical_names = sorted(
        path.name for path in model_students_dir.iterdir() if path.is_dir()
    )
    if not canonical_names:
        return df

    canonical_name_set = set(canonical_names)

    def canonical_name(value):
        value = str(value)
        value_path = Path(value)
        if value_path.name.startswith("seed") and value_path.parent.name in canonical_name_set:
            return value_path.parent.name
        if value in canonical_name_set:
            return value

        # Existing result files may contain arbitrary run suffixes. Match the
        # longest saved experiment name instead of assuming a suffix format.
        matches = [name for name in canonical_names if name in value]
        return max(matches, key=len) if matches else value

    result = df.copy()
    result["EXPERIMENT"] = result["EXPERIMENT"].map(canonical_name)
    return result


def average_across_seeds(df, model_students_dir=None):
    """
    Average SCORE across SEED for every unique experimental setting.
    Keeps all non-numeric identifier columns except SEED and SCORE.
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.upper()
    df = canonicalize_experiment_names(df, model_students_dir)

    if "SEED" not in df.columns:
        raise ValueError("Expected a 'SEED' column, but it was not found.")
    if "SCORE" not in df.columns:
        raise ValueError("Expected a 'SCORE' column, but it was not found.")

    # Clean metric column for later filtering/plotting
    if "METRIC" in df.columns:
        df["METRIC_CLEAN"] = df["METRIC"].astype(str).str.lower().str.strip()
    else:
        raise ValueError("Expected a 'METRIC' column, but it was not found.")

    # Few-shot count is coupled to the benchmark, so it must not split groups.
    group_cols = [
        c for c in df.columns if c not in {"SCORE", "SEED", "NUM_FEWSHOT"}
    ]

    avg_df = (
        df.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            SCORE=("SCORE", "mean"),
            N_SEEDS=("SEED", "nunique"),
        )
    )

    return avg_df


def save_experiment_plots(df, output_root):
    # 1. Setup Styling
    sns.set_theme(style="whitegrid")

    # 2. Pre-processing
    df = df.copy()
    df.columns = df.columns.str.strip().str.upper()
    if "METRIC_CLEAN" not in df.columns:
        df["METRIC_CLEAN"] = df["METRIC"].astype(str).str.lower().str.strip()

    print(f"Generating plots in: {output_root}")

    higher_is_better_metrics = ["acc", "acc_norm", "accuracy", "exact_match", "f1"]

    # --- 1. INDIVIDUAL TASK PLOTS ---
    acc_df = df[df["METRIC_CLEAN"].isin(higher_is_better_metrics)].copy()

    if not acc_df.empty:
        # If a task has acc_norm, drop acc for that task
        tasks_with_acc_norm = acc_df[acc_df["METRIC_CLEAN"] == "acc_norm"]["TASK"].unique()
        acc_df = acc_df[
            ~((acc_df["TASK"].isin(tasks_with_acc_norm)) & (acc_df["METRIC_CLEAN"] == "acc"))
        ]

        for task in acc_df["TASK"].unique():
            for metric in acc_df[acc_df["TASK"] == task]["METRIC_CLEAN"].unique():
                plot_df = acc_df[
                    (acc_df["TASK"] == task) & (acc_df["METRIC_CLEAN"] == metric)
                ]
                if plot_df.empty:
                    continue

                safe_task = f"{task}_{metric}".replace("/", "_").replace(" ", "_")
                task_dir = output_root / "Scores_Per_Task" / safe_task
                task_dir.mkdir(parents=True, exist_ok=True)

                plt.figure(figsize=(12, 7))
                ax = sns.barplot(
                    data=plot_df.sort_values("SCORE", ascending=False),
                    x="EXPERIMENT",
                    y="SCORE",
                    palette="viridis",
                )

                add_labels(ax, fmt="%.4f")

                plt.title(f"{metric.upper()}: {task}")
                plt.xticks(rotation=45, ha="right")
                plt.ylabel("Score")
                plt.tight_layout()
                plt.savefig(task_dir / "scores.png")
                plt.close()

    # --- 2. AVERAGE SCORE (Aggregated) ---
    if not acc_df.empty:
        avg_acc_dir = output_root / "Average_Score"
        avg_acc_dir.mkdir(parents=True, exist_ok=True)

        avg_df = (
            acc_df.groupby(["EXPERIMENT"], as_index=False)
            .agg(
                SCORE=("SCORE", "mean"),
                N_SEEDS=("N_SEEDS", "max"),
            )
        )

        if not avg_df.empty:
            plt.figure(figsize=(12, 7))
            ax = sns.barplot(
                data=avg_df.sort_values("SCORE", ascending=False),
                x="EXPERIMENT",
                y="SCORE",
                palette="mako",
            )

            add_labels(ax, fmt="%.4f")

            plt.title("Average Score Across All Tasks")
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Mean Score")
            plt.tight_layout()
            plt.savefig(avg_acc_dir / "average.png")
            plt.close()

    # --- 3. PERPLEXITY ---
    ppl_df = df[df["METRIC_CLEAN"] == "perplexity"].copy()

    if not ppl_df.empty:
        ppl_dir = output_root / "Perplexity"
        ppl_dir.mkdir(parents=True, exist_ok=True)

        for task in ppl_df["TASK"].unique():
            task_plot_df = ppl_df[ppl_df["TASK"] == task].sort_values(
                "SCORE", ascending=True
            )

            plt.figure(figsize=(12, 7))
            ax = sns.barplot(
                data=task_plot_df,
                x="EXPERIMENT",
                y="SCORE",
                palette="rocket",
            )

            add_labels(ax, fmt="%.3f")

            plt.title(f"Perplexity: {task}")
            plt.ylabel("Perplexity (Lower is Better)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            safe_task = str(task).replace("/", "_").replace(" ", "_")
            plt.savefig(ppl_dir / f"{safe_task}.png")
            plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate labeled plots from a Parquet benchmark file."
    )
    parser.add_argument(
        "input", type=str, help="Path to the .parquet file containing the data"
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()

    if not input_path.exists():
        print(f"Error: File '{input_path}' not found.")
        sys.exit(1)

    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        print(f"Error reading Parquet file: {e}")
        sys.exit(1)

    output_dir = input_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Average across SEED
    model_students_dir = input_path.parent.parent / "models" / "students"
    avg_df = average_across_seeds(df, model_students_dir=model_students_dir)

    # 2) Save averaged dataframe
    avg_data_dir = output_dir / "Averaged_Data"
    save_dataframe_exports(avg_df, avg_data_dir, "average_across_seeds")

    # 3) Plot from averaged dataframe
    save_experiment_plots(avg_df, output_dir)

    print(f"Done. Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
