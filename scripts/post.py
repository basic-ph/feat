
import csv
import matplotlib.pyplot as plt
import numpy as np


storage = []
with open('rve_2020-03-24T174738.csv', 'r', newline='') as file:
    reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        storage.append(row)

rve_1_r = [l[1] for l in storage if l[0] == 1]
rve_1_E2 = [l[5] for l in storage if l[0] == 1]
rve_2_r = [l[1] for l in storage if l[0] == 2]
rve_2_E2 = [l[5] for l in storage if l[0] == 2]

refined_storage = [
    [1, 7.2360125455826765, 169.1960679013824],  # not validated
    [2, 10.233267079464886, 169.3692911969456],  # validated
]
rve_1_mean_E2 = refined_storage[0][2]
rve_1_threshold = 0.05 *  rve_1_mean_E2 # 5% of the mean
rve_1_up_lim = rve_1_mean_E2 + rve_1_threshold
rve_1_low_lim = rve_1_mean_E2 - rve_1_threshold

rve_2_mean_E2 = refined_storage[1][2]
rve_2_threshold = 0.05 *  rve_2_mean_E2 # 5% of the mean
rve_2_up_lim = rve_2_mean_E2 + rve_2_threshold
rve_2_low_lim = rve_2_mean_E2 - rve_2_threshold

# PLOT
plt.ioff()
x = np.linspace(0, 10, 11)

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False)

ax1.plot(rve_1_r, rve_1_E2, ".", label='RVE 1')  # Plot some data on the axes.
ax1.axhline(rve_1_mean_E2, xmin=0, xmax=10, linestyle="--")
ax1.fill_between(x, rve_1_up_lim, rve_1_low_lim, alpha=0.2)
ax1.set_ylim(rve_1_mean_E2 - 0.10 * rve_1_mean_E2, rve_1_mean_E2 + 0.10 * rve_1_mean_E2)
ax1.set_title('TITLE')
ax1.legend()  # Add a legend.

ax2.plot(rve_2_r, rve_2_E2, ".", label='RVE 2')  # Plot some data on the axes.
ax2.axhline(rve_2_mean_E2, xmin=0, xmax=10, linestyle="--")
ax2.fill_between(x, rve_2_up_lim, rve_2_low_lim, alpha=0.2)
ax2.set_ylim(rve_2_mean_E2 - 0.10 * rve_2_mean_E2, rve_2_mean_E2 + 0.10 * rve_2_mean_E2)
ax1.set_title('TITLE')
ax2.legend()  # Add a legend.

plt.show()

