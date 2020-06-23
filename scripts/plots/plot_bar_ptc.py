import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# assembly_pct_time = [2.449/185.247, 2.201/2.547, 0.119/0.441]
# dirichletBC_pct_time = [0.232/185.427, 0.036/2.547, 0.035/0.441]
# solve_pct_time = [182.686/185.427, 0.261/2.547, 0.270/0.441]


# mpl.style.use("seaborn")

labels = ["assembly", "dirichlet BC", "solve"]

base_pct = [2.449/185.247, 0.232/185.427, 182.686/185.427]
sparse_pct = [2.201/2.547, 0.036/2.547, 0.261/2.547]
vector_pct = [0.119/0.441, 0.035/0.441, 0.270/0.441]


x = np.arange(len(labels))  # label locations
width = 0.15  # width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, base_pct, width, label="base")
rects2 = ax.bar(x, sparse_pct, width, label="sparse")
rects3 = ax.bar(x + width, vector_pct, width, label="vector")

ax.set_ylabel("% of total time")
ax.set_xlabel("FEM phase")
ax.set_xticks(x)
ax.set_xticklabels(labels)
# ax.grid(axis="y")
ax.legend()

fig.tight_layout()

plt.show()