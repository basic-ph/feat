import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.3f}%\n({:.3f} s)".format(pct, absolute)

def plot_pie(name, total_time, times, phases):
    fig, ax = plt.subplots(figsize=(8, 4), subplot_kw=dict(aspect="equal"))
    wedges, _ = ax.pie(times)

    ax.legend(
        wedges,
        phases,
        title=f"{name}: {total_time}",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    # ax.set_title("Matplotlib bakery: A pie")

mpl.style.use("seaborn")
plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    # "text.usetex": True,     # use inline math for ticks
    # "pgf.rcfonts": False     # don't setup fonts from rc parameters
    })

complete_data = {
    "base": {
        "name": "base_analysis",
        "total_time": "185.427 s",
        "times": [
            182.686,
            2.449,
            0.232,
            0.06,
        ],
        "phases": [
            "solve: 182.686 s",
            "assembly: 2.449 s",
            "dirichlet BC: 0.232 s",
            "other: 0.06 s",
        ]
    },
    "sparse": {
        "name": "sp_base_analysis",
        "total_time": "2.547 s",
        "times": [
            0.261,
            2.201,
            0.036,
            0.049,
        ],
        "phases": [
            "solve: 0.261 s",
            "assembly: 2.201 s",
            "dirichlet BC: 0.036 s",
            "other: 0.049 s",
        ]
    },
    "vector": {
        "name": "vector_analysis",
        "total_time": "0.441 s",
        "times": [
            0.270 / 0.441,
            0.119 / 0.441,
            0.035 / 0.441,
            0.017 / 0.441,
        ],
        "phases": [
            "solve: 0.270 s",
            "assembly: 0.119 s",
            "dirichlet BC: 0.035 s",
            "other: 0.017 s",
        ]
    },
}

print(plt.style.available)
for key,value in complete_data.items():
    plot_pie(
        value["name"],
        value["total_time"],
        value["times"],
        value["phases"],
    )

plt.show()