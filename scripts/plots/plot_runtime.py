import matplotlib.pyplot as plt



num_nodes_data = [8826, 17820, 35327, 141282, 565549]
num_dofs_data = [n*2 for n in num_nodes_data]
num_dofs_short = [n*2 for n in num_nodes_data[:2]]

time_base_data = [
    round(t, 3)
    for t in [22.343284845352173 , 184.2389760017395]
]
time_sparse_data = [
    round(t, 3)
    for t in [
        1.0877270698547363,
        2.205213785171509,
        4.466970205307007,
        19.1698796749115,
        83.22076272964478,
    ]
]
time_vector_data = [
    round(t, 3)
    for t in [
        0.25047993659973145,
        0.5708961486816406,
        1.258349895477295,
        5.730280876159668,
        29.83337163925171,
    ]
]

x0 = 8826 * 2
y0 = 1.0877270698547363

y_sparse = [(y0/x0)*n for n in num_dofs_data]

x1 = 8826 * 2
y1 = 0.25047993659973145
y_vector = [(y1/x1)*n for n in num_dofs_data]

# PLOT ---------------------------------------------------------------------------------
fig, ax = plt.subplots()

ax.plot(num_dofs_short, time_base_data,"o-C0", label="base")
ax.plot(num_dofs_data, time_sparse_data,"o-C1", label="sparse")
ax.plot(num_dofs_data, time_vector_data,"o-C2", label="vector")

# linear functions
# ax.plot(num_dofs_data, y_sparse, "--C1", label="void")
# ax.plot(num_dofs_data, y_vector, "--C2", label="void")

ax.set(
    xlabel='number of dofs',
    ylabel='Time [s]',
    title='Execution Time'
)
ax.grid()
ax.set_yscale('log')
ax.set_xscale('log')
fig.legend()

plt.show()