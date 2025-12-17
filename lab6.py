import numpy
import numpy as np
import scipy
import matplotlib.pyplot as plt

#1.
#a.
t_a1 = np.arange(-3, 4, 1/100)
x_a1 = numpy.sinc(t_a1)

# plt.figure(figsize=(10, 5))
# plt.plot(t, x)
# plt.grid(True)
# plt.show()

#b.
t_a2 = np.arange(-3, 4)
x_a2 = numpy.sinc(t_a2)

x_a3 = numpy.sinc(t_a1) ** 2

fig3, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(t_a1, x_a1 ** 2, color='k')
axes[0].plot(t_a1, x_a1, linestyle='--', color='green')
axes[0].set_xlim(-3, 3)
axes[0].set_ylim(-0.5, 1.5)
axes[0].axhline(0, color='k', linestyle='-', linewidth=1.5)

for i, j in enumerate(t_a2):
    print(i,j)
    index_marker = j
    axes[0].plot(index_marker, x_a2[i], 'ro', markersize=8)
    axes[0].plot([index_marker, index_marker], [0, x_a2[i]], 'r-', linewidth=1)

axes[1].plot(t_a1, x_a1 ** 2, color='k')
axes[1].plot(t_a1, x_a3, linestyle='--', color='green')
axes[1].set_xlim(-3, 3)
axes[1].set_ylim(-0.5, 1.5)
axes[1].axhline(0, color='k', linestyle='-', linewidth=1.5)

plt.tight_layout()
plt.show()