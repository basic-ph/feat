import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl



num_dofs = [17652, 35640, 70654, 282564, 1131098]
sparse_assembly = [1.060, 2.165, 4.199, 16.984, 69.201]
vector_assembly = [0.066, 0.104, 0.226, 1.017, 4.266]

assebly_speedup = [round(item[0]/item[1],2) for item in zip(sparse_assembly, vector_assembly)]
print(assebly_speedup)  # [16.06, 20.82, 18.58, 16.7, 16.22]

mpl.style.use("seaborn")
plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    # "text.usetex": True,     # use inline math for ticks
    # "pgf.rcfonts": False     # don't setup fonts from rc parameters
})

x = np.arange(len(num_dofs))  # label locations
width = 0.35  # width of the bars

fig, ax = plt.subplots()
rects2 = ax.bar(x, assebly_speedup, width, label="vectorized assembly $S_p$")

ax.set_ylabel("$S_p$")
ax.set_xlabel("DOFs number")
ax.set_xticks(x)
ax.set_xticklabels(num_dofs)
# ax.grid(axis="y")
ax.legend()

fig.tight_layout()

plt.show()