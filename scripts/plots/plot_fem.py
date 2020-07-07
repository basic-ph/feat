import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.style.use("seaborn")
plt.rcParams.update({
"font.family": "serif",  # use serif/main font for text elements
# "text.usetex": True,     # use inline math for ticks
# "pgf.rcfonts": False     # don't setup fonts from rc parameters
})

# -- 2020-07-06T13-48-58.csv --
# 0,4,0,30.0,14411,12.19444074409448,0
# 0,4,1,30.0,50075,12.040223020423582,0
# 0,4,2,30.0,201895,11.974549104271635,0
# 0,4,3,30.0,794172,11.956825861113327,1
#
# side = 30, Vf = 60%


num_nodes = [14411, 50075, 201895, 794172]
num_dofs = [2*n for n in num_nodes]

E2 = [12.194, 12.04, 11.975, 11.957]

fig, ax = plt.subplots()

ax.plot(num_dofs, E2, "o--")

ax.set(
    xlabel="DOFs number",
    ylabel="$E_2$ [GPa]"
)
ax.set_ylim(11.93, 12.23)
# ax.set_xscale('log')

# fig.tight_layout()
plt.show()