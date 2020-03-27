import pprint
import csv
import matplotlib.pyplot as plt
import numpy as np


storage = []
with open("./data/output/rve_b.csv", 'r', newline='') as file:
    reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        storage.append(row)

realizations = 10  # number of realizations to compare
i_max = int(storage[-1][0]+1)  # rve size number of the last data row
print(i_max)

# fig, axs = plt.subplots(1, i_max)
x = np.linspace(0, 11, 12)

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
    
    # ax = axs[i]
    fig, ax = plt.subplots()
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
    ax.set_title(f'RVE #{i+1} Convergence Analysis')

plt.show()