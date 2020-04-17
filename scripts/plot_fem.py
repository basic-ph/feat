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
    parser.add_argument("i", help="rve id")
    parser.add_argument("r", help="realization id")
    args = parser.parse_args()

    raw_data = []
    with open(args.data_path, 'r', newline='') as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            raw_data.append(row)
    
    realizations = 30  # number of realizations to compare
    i_max = int(raw_data[-1][0]+1)  # rve size number of the last data row
    data = []

    i = float(args.i)
    r = float(args.r)
    dof_data = [item[4]*2 for item in raw_data if item[:2] == [i, r]]  # nodes * 2
    E2_data = [item[5] for item in raw_data if item[:2] == [i, r]]
    
    print(dof_data)
    print(E2_data)

    fig, ax = plt.subplots()

    ax.plot(dof_data, E2_data, "o-")
    ax.set_xlabel("dof number")
    ax.set_ylabel(r"$E2 \quad[unit]$")
    ax.set_title('Mesh refinement analysis')
    fig.text(0.5, 0.8, f'$ i={i},\quad r={r}$', fontsize="large")

    plt.show()



if __name__ == "__main__":
    main()