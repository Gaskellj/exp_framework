"""
Simple visualization script assuming parallelism with results in a single CSV file.
Plots either the best fitness per generation, the average best fitness so far, or both.

The CSV file format is as follows:
    col 0: generation number (used to group runs)
    col 1: best fitness in this generation
    col 2: best fitness so far
    cols > 2: genome values

Author: James Gaskell
April 9th, 2025
"""

import csv
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def process_csv(filename):
    """
    Reads a CSV file and extracts both fitness datasets.

    Args:
        filename (str): Path to the CSV file

    Returns:
        float: Group size (based on number of rows with generation == 0)
        list: Average best fitness values per generation
        list: Best fitness per generation (minimum)
        list: Lower quartiles (for average best fitness)
        list: Upper quartiles (for average best fitness)
    """
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)

    generation_data = defaultdict(list)
    group_size = 0

    for row in data[1:]:
        generation = row[0]
        value = float(row[2])
        generation_data[generation].append(value)

        # Count how many rows have generation == 0
        if generation == '0':
            group_size += 1

    avg_values = []
    best_values = []
    lower_quantiles = []
    upper_quantiles = []
    percentiles = [25, 75]

    for values in generation_data.values():
        values = np.array(values)
        avg_values.append(np.mean(values))
        best_values.append(np.min(values))
        lower_quantiles.append(np.percentile(values, percentiles[0]))
        upper_quantiles.append(np.percentile(values, percentiles[1]))

    return group_size, avg_values, best_values, lower_quantiles, upper_quantiles


def plot_fitness(group_size, avg_values, best_values, lower_quantiles, upper_quantiles, mode):
    """
    Plots the fitness data using matplotlib.

    Args:
        group_size (float): Average group size of parallel runs
        avg_values (list): Average best fitness per generation
        best_values (list): Best fitness per generation
        lower_quantiles (list): 25th percentile values for average best fitness
        upper_quantiles (list): 75th percentile values for average best fitness
        mode (str): "best" for best fitness per generation, "avg" for average best fitness so far, "both" for both
    """
    x_values = np.arange(1, len(avg_values) + 1)

    avg_color = '#ff871d'  # Orange for average best fitness
    best_color = '#4fafd9'  # Blue for best fitness per generation
    shade_color = '#ffb266'

    plt.figure(figsize=(8, 6))

    if mode in ("avg", "both"):
        plt.plot(x_values, avg_values, marker='o', linestyle='-', color=avg_color, markersize=4, label="Average Best Fitness")
        plt.fill_between(x_values, lower_quantiles, upper_quantiles, color=shade_color, alpha=0.2, label="25th-75th Percentile")

    if mode in ("best", "both"):
        plt.plot(x_values, best_values, marker='o', linestyle='-', color=best_color, markersize=4, label="Best Fitness Per Parallel Generation")

    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    
    title_mode = {
        "best": "Best Fitness per Generation",
        "avg": "Average Best Fitness So Far",
        "both": "Best and Average Fitness per Generation"
    }
    
    plt.title(f'{title_mode[mode]} for {int(group_size)} Parallel Runs', fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--mode", type=str, default="both",
                        help="best: best fitness per parallel generation, avg: average best fitness so far, both: both")
    
    parser.add_argument("--filename", type=str, default="latest.csv",
                        help="Path to the CSV file. Defaults to the most recent CSV file in the directory.")

    args = parser.parse_args()
    
    group_size, avg_values, best_values, lower_quartiles, upper_quantiles = process_csv(args.filename)
    plot_fitness(group_size, avg_values, best_values, lower_quartiles, upper_quantiles, args.mode)

