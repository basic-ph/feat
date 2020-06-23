import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl



num_dofs_data = [17652, 35640, 70654, 282564, 1131098]

time_base = [22.34, 184.24]
time_sparse = [1.09, 2.21, 4.47, 19.17, 83.22]
time_vector = [0.25, 0.57, 1.26, 5.73, 29.83]

vector_sparse_speed = [round(item[0]/item[1],2) for item in zip(time_sparse, time_vector)]
print(vector_sparse_speed)  # [4.36, 3.88, 3.55, 3.35, 2.79]
print()
sparse_base_speed = [round(item[0]/item[1],2) for item in zip(time_base, time_sparse[:2])]
print(sparse_base_speed)  # [20.5, 83.37]
print()
vector_base_speed = [round(item[0]/item[1],2) for item in zip(time_base, time_vector[:2])]
print(vector_base_speed)  # [89.36, 323.23]

mpl.style.use("seaborn")
plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    # "text.usetex": True,     # use inline math for ticks
    # "pgf.rcfonts": False     # don't setup fonts from rc parameters
})

x = np.arange(2)
y = np.arange(5)
width = 0.15

fig, ax = plt.subplots()
rect1 = ax.bar(x - width, sparse_base_speed, width, label="S_p sparse (base)")
rect2 = ax.bar(x, vector_base_speed, width, label="$S_p$ vector (base)")
rect3 = ax.bar(y + width, vector_sparse_speed, width, label="$S_p$ vector (sparse)")

ax.set_ylabel("$S_p$")
ax.set_xlabel("DOFs number")
ax.set_xticks(y)
ax.set_xticklabels(num_dofs_data)
# ax.grid(axis="y")
ax.legend()

fig.tight_layout()

plt.show()