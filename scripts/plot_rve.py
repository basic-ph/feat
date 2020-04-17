import argparse
import csv
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


"""
[i, r, s, side, nodes, E2, E2_mean] data composition
"""

def main():
    parser = argparse.ArgumentParser(
        prog="python post.py",
    )
    parser.add_argument(
        "data_path",
        help="path to the data file used to create plots"
    )
    args = parser.parse_args()

    raw_data = []
    with open(args.data_path, 'r', newline='') as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            raw_data.append(row)
    
    realizations = 30  # number of realizations to compare
    i_max = int(raw_data[-1][0]+1)  # rve size number of the last data row
    data = []

    for i in range(i_max):  # i_max
        print(i)
        tmp_E2_data = []  # list containing all refined E2 for different realizations of the same RVE
        for r in range(realizations):
            # takes the last item (maximus s value) of data regarding rve #i and realiz #r
            E2_refined = [item[5] for item in raw_data if item[:2] == [i, r]][-1]
            tmp_E2_data.append(E2_refined)
        data.append(tmp_E2_data)

    # create RVE convergence plot
    fig1, ax1 = plt.subplots()
    ax1.boxplot(data, whis=(0,100), meanline=True, showmeans=True)
    ax1.set_xlabel("RVE id")
    ax1.set_ylabel(r"$\overline{E2} \quad[unit]$")
    ax1.set_title(f'RVE Convergence Analysis')
    # ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # fig1.text(0.5, 0.8, '$V_f = 15\%$', fontsize="large")

    plt.show()


if __name__ == "__main__":
    main()