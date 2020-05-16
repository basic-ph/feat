import argparse
import csv
import math

import matplotlib.pyplot as plt
import numpy as np


def main():
    # [
    #   0 s: id del sample,
    #   1 n: numero di fibre nel dominiio,
    #   2 coarse_cl,
    #   3 side,
    #   4 num_nodes,
    #   5 E2,
    #   6 {0|1}
    # ]
    # TODO find latest csv file using name and use as default data_path
    parser = argparse.ArgumentParser(
        prog="python plot_rve.py",
    )
    parser.add_argument(
        "data_path",
        help="path to the data file used to create plots"
    )
    args = parser.parse_args()

    data = []
    with open(args.data_path, 'r', newline='') as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            data.append(row)
    
    data = [item for item in data if item[6] == 1.0]  # select only converged values (last value = 1)
    samples = int(data[-1][0] + 1)  # infers the number of samples from the last entry
    steps = [item[1] for item in data if item[0] == data[-1][0]]  # different-sized domanis computed for each sample
    print(data)
    print(samples)
    print(steps)

    fig, ax = plt.subplots()
    for s in range(samples):
        n_data = [item[1] for item in data if item[0] == float(s)]
        E2_data = [item[5] for item in data if item[0] == float(s)]
        ax.plot(n_data, E2_data, "--", label=f"sample {s+1}")
    
    max_data = []
    min_data = []
    for n in steps:
        E2_data = [item[5] for item in data if item[1] == n]
        E2_max = max(E2_data)
        E2_min = min(E2_data)
        max_data.append(E2_max)
        min_data.append(E2_min)

    ax.fill_between(steps, max_data, min_data, alpha=0.3)
    # ax.plot(steps, max_data, ".")
    # ax.plot(steps, min_data, ".")
    
    ax.set(
        xlabel='number of fibers in the domain',
        ylabel='$E_2$ [GPa]',
        title='RVE Convergence'
    )
    ax.grid()
    fig.legend()

    plt.show()

if __name__ == "__main__":
    main()