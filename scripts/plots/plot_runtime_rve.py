import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def sec_to_min(seconds):
    return seconds / 60

def min_to_sec(minutes):
    return minutes * 60


num_fibers = [100, 300, 500]
sparse_times = [820.0161843299866, 2638.3441269397736, 4797.735440969467]
vector_times = [434.98689222335815, 1528.1057493686676, 2884.404346227646]

# Vf = 30%, 5 samples, 10 steps

mpl.style.use("seaborn")
plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    # "text.usetex": True,     # use inline math for ticks
    # "pgf.rcfonts": False     # don't setup fonts from rc parameters
})

fig, ax = plt.subplots()
ax.plot(num_fibers, sparse_times, "o-", label="sparse")
ax.plot(num_fibers, vector_times, "o-", label="vector")

sec_y = ax.secondary_yaxis('right', functions=(sec_to_min, min_to_sec))
sec_y.set_ylabel('Time [min]')

ax.legend()
ax.set(
    xlabel='number of fibers in the support square',
    ylabel='Time [s]',
)
fig.tight_layout()
plt.show()