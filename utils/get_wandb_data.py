#!/bin/python

import os
import re
import wandb
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


timestamp = "correct_22_04_25"
pickle_output_file = f"next_token_data_{timestamp}.pkl"
if not os.path.exists(pickle_output_file):
    api = wandb.Api()
    project_name = "next-token-failures-waypoint-correct"

    # Fetch all runs in the specified project
    runs = api.runs(path=project_name)
    print("Total runs in the project:", len(runs))

    # Iterate through each run and retrieve the data
    summary_list, config_list, name_list = [], [], []
    data_list = []
    for run in runs:
        summary_list.append(run.summary._json_dict)
        config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})
        name_list.append(run.name)

        rows = []
        for row in run.scan_history():
            rows.append(row)
        history_df = pd.DataFrame(rows)
        print("Columns for run:", run.name, "\n", history_df.columns)

        data_dict = {}
        for metric_key in ["test/accuracy", "test/forced loss", "test/forced accuracy"]:
            print("Retrieving key:", metric_key)
            assert metric_key in history_df.columns, f"{metric_key} not in {history_df.columns}"
            time_series = history_df[[metric_key]].dropna()
            print("TS shape:", time_series.shape)
            data_dict[metric_key.replace("/", "_")] = time_series
        data_list.append(data_dict)

    output_dict = dict(summary_list=summary_list, config_list=config_list, name_list=name_list, data_list=data_list)
    with open(pickle_output_file, "wb") as f:
        pickle.dump(output_dict, f)

# Reload the file
print("Loading data from file:", pickle_output_file)
with open(pickle_output_file, "rb") as f:
    output_dict = pickle.load(f)
print("Loaded keys:", output_dict.keys())
name_list = output_dict["name_list"]
summary_list = output_dict["summary_list"]
data_list = output_dict["data_list"]
print("Model names:", name_list)

output_file_format = "pdf"
plots_output_dir = f"plots_waypoint_{timestamp}/"
if not os.path.exists(plots_output_dir):
    os.mkdir(plots_output_dir)
    print("Plots output directory created:", plots_output_dir)

# Plot the predictions
waypoint_dict = {}
for idx, full_name in enumerate(name_list):
    acc = data_list[idx]["test_accuracy"].to_numpy()
    assert acc.shape[1] == 1, acc.shape

    # Get run attributes
    waypoint_len = int(full_name.split("_waypoint_len_")[1].split("_")[0])
    print("Run:", full_name)
    print("Waypoint len:", waypoint_len)

    heads, boundary_cond, weights = None, None, None
    if "_heads_" in full_name:
        heads = full_name.split("_heads_")[1].split("_")[0]
        weights = full_name.split("_heads_")[1].split("_")[2]
        boundary_cond = full_name.split("_boundary_")[1].split("_")[0]

    if waypoint_len not in waypoint_dict:
        waypoint_dict[waypoint_len] = {}
    current_key = f"H: {heads} / B: {boundary_cond}" if heads is not None else "Default"
    if current_key not in waypoint_dict[waypoint_len]:
        waypoint_dict[waypoint_len][current_key] = []
    waypoint_dict[waypoint_len][current_key].append(acc[:, 0])

# Plot the predictions
assert len(waypoint_dict.keys()) == 4, waypoint_dict.keys()
fig, ax = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)  # 4 waypoints
fontsize = 18
plot_conf_interval = True
for idx, waypoint_len in enumerate(waypoint_dict.keys()):
    plot_x = idx % 2
    plot_y = idx // 2

    for model_config in waypoint_dict[waypoint_len].keys():
        val_list = np.stack(waypoint_dict[waypoint_len][model_config], axis=0)
        print("Val list:", val_list.shape)
        mean, std = np.mean(val_list, axis=0), np.std(val_list, axis=0)
        x = list(range(1, len(mean)+1))
        ax[plot_x, plot_y].plot(x, mean, label=model_config, linewidth=4, alpha=0.7)
        if plot_conf_interval:
            standard_error = std / np.sqrt(val_list.shape[0])  # standard err = standard dev / sqrt(n)
            min_range = mean - 1.96 * standard_error
            max_range = mean + 1.96 * standard_error
        else:
            min_range = mean - std
            max_range = mean + std
        ax[plot_x, plot_y].fill_between(x, min_range, max_range, alpha=0.1)

    ax[plot_x, plot_y].set_ylim(0., 100.)
    if idx == 3:
        ax[plot_x, plot_y].legend(loc="lower left", fontsize=fontsize-4)
    ax[plot_x, plot_y].set_title(f"Waypoint len: {waypoint_len}", fontsize=fontsize)

    ax[plot_x, plot_y].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[plot_x, plot_y].tick_params(axis='both', which='minor', labelsize=fontsize)

fig.supxlabel('Training epochs', fontsize=fontsize)
fig.supylabel('Accuracy (%)', fontsize=fontsize)

plt.tight_layout()
output_file = os.path.join(plots_output_dir, f"graph_waypoint_task{'_conf_int' if plot_conf_interval else ''}.{output_file_format}")
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close()

print("Plotting finished")
