import json
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import pandas as pd


pfin = Path("/home/ampere/vanroy/scandeval/scandeval_benchmark_results.jsonl")
dout = "/home/local/vanroy/dutch-instruction-datasets/benchmark_output"
data = [json.loads(line) for line in pfin.read_text().splitlines() if line.strip()]
pprint(data)
groups = defaultdict(list)

dataset2model_names = defaultdict(set)
for d in data:
    model_name = d["model"].split("/")[-1]
    model_name = "fietje-2b-ckpt-7185" if model_name == "fietje-2b" else model_name
    dataset_name = d["dataset"]

    if model_name in dataset2model_names[dataset_name]:
        continue

    dataset2model_names[dataset_name].add(model_name)

    info = {
        "model": model_name,
        "task": d["task"],
        "dataset": dataset_name,
    }
    info = {**info, **d["results"]["total"]}
    groups[dataset_name].append(info)

groups = {k: pd.DataFrame(v) for k, v in groups.items()}

aggrs = {
    "dutch-social": "test_macro_f1",
    "conll-nl": "test_micro_f1",
    "scala-nl": "test_macro_f1",
    "squad-nl": "test_f1",
    "wiki-lingua-nl": "test_bertscore",
    "mmlu-nl": "test_accuracy",
    "hellaswag-nl": "test_accuracy",
}

avg_df = pd.DataFrame()
for dataset, aggr_col in aggrs.items():
    dataset_df = groups[dataset].copy()
    dataset_df = dataset_df[["model", aggr_col]]
    dataset_df.rename(columns={aggr_col: f"{dataset}_{aggr_col}"}, inplace=True)

    if avg_df.empty:
        avg_df = dataset_df
    else:
        avg_df = pd.merge(avg_df, dataset_df, on="model", how="inner")

avg_df["average"] = avg_df.mean(axis=1, numeric_only=True)
avg_df["average_no_conll"] = avg_df[[col for col in avg_df.columns if "conll" not in col]].mean(
    axis=1, numeric_only=True
)

groups["average"] = avg_df
with pd.ExcelWriter(f"{dout}/benchmark_results.xlsx", engine="xlsxwriter") as writer:
    for group, group_df in groups.items():
        group_df.to_excel(writer, sheet_name=group, index=False)


pprint(groups)

# PLOTS


# Identify test columns
for training_type in ("", "-sft"):
    for group, group_df in groups.items():
        group_df = group_df[group_df["model"].str.startswith(f"fietje-2b{training_type}-ckpt-")].copy()
        group_df["checkpoint"] = group_df["model"].str.replace(f"fietje-2b{training_type}-ckpt-", "").astype(int)
        test_columns = [col for col in group_df.columns if col.startswith("test_") and not col.endswith("_se")]
        if training_type == "-sft":
            training_type_title = " (SFT)"
            output_dir = f"{dout}/sft"
        else:
            training_type_title = ""
            output_dir = f"{dout}/cpt"

        Path(output_dir).mkdir(exist_ok=True, parents=True)

        for test_col in test_columns:
            se_col = test_col + "_se"  # Standard error column name

            plt.figure(figsize=(10, 6))
            plt.errorbar(
                group_df["checkpoint"],
                group_df[test_col],
                yerr=group_df[se_col],
                fmt="o-",
                capsize=5,
                label="Score with SE",
            )
            plt.title(f"Progression of {test_col} over Checkpoints{training_type_title}")
            plt.xlabel("Checkpoint")
            plt.ylabel(test_col)
            plt.xticks(group_df["checkpoint"], labels=group_df["model"])
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{group}_{test_col}.png")
            plt.close()

# Plot averages

for training_type in ("", "-sft"):
    for group, group_df in groups.items():
        group_df = group_df[group_df["model"].str.startswith(f"fietje-2b{training_type}-ckpt-")].copy()
        group_df["checkpoint"] = group_df["model"].str.replace(f"fietje-2b{training_type}-ckpt-", "").astype(int)
        if training_type == "-sft":
            training_type_title = " (SFT)"
            output_dir = f"{dout}/sft"
        else:
            training_type_title = ""
            output_dir = f"{dout}/cpt"

        for test_column in ("average_no_conll", "average"):
            plt.figure(figsize=(10, 6))
            try:
                plt.errorbar(group_df["checkpoint"], group_df[test_column], fmt="o-", capsize=5, label="Score")
            except KeyError:
                continue
            plt.errorbar(group_df["checkpoint"], group_df[test_column], fmt="o-", capsize=5, label="Score")
            plt.title(f"Progression of average score over Checkpoints")
            plt.xlabel("Checkpoint")
            plt.ylabel(test_column)
            plt.xticks(group_df["checkpoint"], labels=group_df["model"])
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{test_column}_score.png")

print(avg_df)
