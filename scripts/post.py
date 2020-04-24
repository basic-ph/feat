import argparse
import csv
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def main():
    parser = argparse.ArgumentParser(
        prog="python post.py",
    )
    parser.add_argument(
        "data_path",
        help="path to the data file used to create plots"
    )
    parser.add_argument(
        "-g",
        "--group",
        help="group up all graphs in the same figure",
        dest="group",
        action="store_true",
    )
    args = parser.parse_args()

    storage = []
    with open(args.data_path, 'r', newline='') as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            storage.append(row)

    realizations = 30  # number of realizations to compare
    i_max = int(storage[-1][0]+1)  # rve size number of the last data row
    if args.group:
        rows = math.ceil(i_max / 2)  # 2 is the num of columns wanted
        fig, axs = plt.subplots(rows, 2, constrained_layout=True)  # figure and axes for validation of each RVE
    x = np.linspace(0, 11, 12)
    mean_E2_data = []

    for i in range(i_max):  # i_max
        print(i)
        E2_data = []
        r_data = []
        for r in range(realizations):
            init = [i, r]
            # takes the last item (maximus s value) of data regarding rve #i and realiz #r
            data = [item for item in storage if item[:2] == init][-1]
            E2_data.append(data[5])
            r_data.append(data[1]+1)  # convert to 1 offset
        mean_E2 = data[6]
        mean_E2_data.append(mean_E2)
        
        if args.group:
            ax = axs.ravel()[i]
        else:
            fig, ax = plt.subplots(constrained_layout=True)
        
        ax.plot(r_data, E2_data, ".r", label=f"RVE #{i}")
        ax.axhline(mean_E2, linestyle="--")
        ax.fill_between(
            x,
            mean_E2 + 0.01*mean_E2,
            mean_E2 - 0.01*mean_E2,
            alpha=0.2
        )
        ax.set_xlabel("realization id")
        ax.set_ylabel("E2 [unit]")
        ax.set_title(f'RVE #{i+1} Validation')

    # create RVE convergence plot
    fig2, ax2 = plt.subplots()
    i_data = list(range(1, i_max+1))
    print(i_data)
    print(mean_E2_data)

    ax2.plot(i_data, mean_E2_data, "o-g")
    ax2.set_xlabel("RVE id")
    ax2.set_ylabel(r"$\overline{E2} \quad[unit]$")
    ax2.set_title('RVE Convergence Analysis')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()


if __name__ == "__main__":
    main()
