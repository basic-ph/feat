import argparse
import csv
import math
import statistics

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


def main():
    # [
    #   0 p: id del sample,
    #   1 s: id dello step (RVE)
    #   2 m: id della mesh,
    #   3 box_side,
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
    with open(args.data_path, 'r', newline='') as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            data.append(row)
    
    data = [item for item in data if item[6] == 1.0]  # select only converged values (last value = 1)
    samples = int(data[-1][0] + 1)  # infers the number of samples from the last entry
    steps = [item[3] for item in data if item[0] == data[-1][0]]  # side of different-sized domanis computed for each sample

    mpl.style.use("seaborn")
    plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    # "text.usetex": True,     # use inline math for ticks
    # "pgf.rcfonts": False     # don't setup fonts from rc parameters
    })
    fig, ax = plt.subplots()
    for s in range(samples):
        side_data = [item[3] for item in data if item[0] == float(s)]
        E2_data = [item[5] for item in data if item[0] == float(s)]
        ax.plot(side_data, E2_data, "--", label=f"sample {s+1}")
    
    # last E2 for each sample, should be the best estimate
    # best_data_side = [steps[-1] for i in range(samples)]
    best_data = [item[5] for item in data if item[1] == len(steps)-1]
    # ax.plot(best_data_side, best_data, "ok", markersize="3")
    print(best_data)
    mean = statistics.mean(best_data)
    sigma = statistics.stdev(best_data)
    
    textstr = (
        f"$\\overline{{E}}_2$ = {mean:.3f} GPa\n"
        f"$\sigma$ = {sigma:.3f} Gpa"
    )
    ax.text(
        0.5,
        0.3,
        textstr,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(facecolor='w')  # boxstyle='round', alpha=0.5
    )


    # Fill area between max and min
    # max_data = []
    # min_data = []
    # for n in steps:
    #     E2_data = [item[5] for item in data if item[3] == n]
    #     E2_max = max(E2_data)
    #     E2_min = min(E2_data)
    #     # max_data.append(E2_max)
        # min_data.append(E2_min)

    # ax.fill_between(steps, max_data, min_data, alpha=0.3)
    # ax.plot(steps, max_data, ".")
    # ax.plot(steps, min_data, ".")
    
    ax.set(
        xlabel='RVE size',
        ylabel='$E_2$ [GPa]',
        # title='RVE Convergence'
    )
    # ax.set_ylim(8.5, 9.5)
    # ax.grid()
    fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()