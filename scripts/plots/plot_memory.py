
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.style.use("seaborn")
plt.rcParams.update({
"font.family": "serif",  # use serif/main font for text elements
# "text.usetex": True,     # use inline math for ticks
# "pgf.rcfonts": False     # don't setup fonts from rc parameters
})



x = np.linspace(0, 100, 1000)

fig, ax = plt.subplots()

ax.plot(x, 32* x**2, label="base")
ax.plot(x, 1152*x, label="sparse")
ax.plot(36, 41472, "o")

ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.set(
    xlabel="$n$ (number of nodes)",
    ylabel="memory usage [bytes]"
)

fig.legend(loc='upper right', bbox_to_anchor=(0.75, 0.90))
fig.tight_layout()

plt.show()