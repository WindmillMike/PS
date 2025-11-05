import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_pdf import PdfPages
import math

#1.
marime = int(input("Marime = "))

def calcElemFourier(linie, coloana, marime, semn):
    grad = semn * 2 * math.pi * linie * coloana / marime
    return complex(math.cos(grad), math.sin(grad)) / math.sqrt(marime)

def matriceaFourier(marime):
    Fourier = np.zeros((marime, marime), dtype=np.complex128)
    for i in range(marime):
        for j in range(marime):
            Fourier[i, j] = calcElemFourier(i, j, marime, -1)
    return Fourier

F = matriceaFourier(marime)

fig, axs = plt.subplots(2, marime, figsize=(20, 5))

for i in range(marime):
    axs[0, i].plot(F[i].real, color='blue')
    axs[0, i].set_title(f"Linia {i} - Real")
    axs[0, i].set_ylim(-1.2, 1.2)

    axs[1, i].plot(F[i].imag, color='red')
    axs[1, i].set_title(f"Linia {i} - Imaginar")
    axs[1, i].set_ylim(-1.2, 1.2)

plt.show()

FH = np.conj(F.T)
prod = FH @ F

identitate = np.eye(marime, dtype=np.complex64)
este_unitara = np.allclose(prod, identitate)
diferenta = np.linalg.norm(prod - identitate)

print(este_unitara)
print(diferenta)


# 2.

# a.
frecventa = 5
frecventaEsantionare = 1000
durata = 1
t = np.arange(0, durata, 1 / frecventaEsantionare)
x = np.sin(2 * np.pi * frecventa * t + 0.5 * np.pi)
esantioane = len(x)

omega = 1
y = x * np.exp(-2j * np.pi * omega * t)

fig1, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].set_title("Semnalul x[n] in Domeniul Timp")
axes[0].set_xlabel("Timp (esantioane)")
axes[0].set_ylabel("Amplitudine")
axes[0].set_xlim([0, esantioane])
axes[0].set_ylim([-1.1, 1.1])
axes[0].axhline(0, color='black', linewidth=0.5)

line_time, = axes[0].plot([], [], color='green', linewidth=1.5)
point_time, = axes[0].plot([], [], 'ro', markersize=8)
line_amp, = axes[0].plot([], [], 'r-', linewidth=1)

axes[1].set_title("Reprezentare in Planul Complex (Infasurare)")
axes[1].set_xlabel("Real")
axes[1].set_ylabel("Imaginar")
axes[1].set_xlim([-1.1, 1.1])
axes[1].set_ylim([-1.1, 1.1])
axes[1].set_aspect('equal', adjustable='box')
axes[1].axhline(0, color='black', linewidth=0.5)
axes[1].axvline(0, color='black', linewidth=0.5)

line_complex, = axes[1].plot([], [], color='slateblue', linewidth=1.5)
point_complex, = axes[1].plot([], [], 'ro', markersize=8)
line_radius, = axes[1].plot([], [], 'r-', linewidth=1)

plt.tight_layout()


def init1():
    line_time.set_data([], [])
    point_time.set_data([], [])
    line_amp.set_data([], [])
    line_complex.set_data([], [])
    point_complex.set_data([], [])
    line_radius.set_data([], [])
    return line_time, point_time, line_amp, line_complex, point_complex, line_radius


def animate1(i):
    step = 10
    end_index = i * step

    if end_index > esantioane:
        end_index = esantioane

    line_time.set_data(np.arange(end_index), x[:end_index])
    point_time.set_data(end_index, x[end_index - 1])
    line_amp.set_data([end_index, end_index], [0, x[end_index - 1]])

    line_complex.set_data(y[:end_index].real, y[:end_index].imag)
    point_complex.set_data(y[end_index - 1].real, y[end_index - 1].imag)
    line_radius.set_data([0, y[end_index - 1].real], [0, y[end_index - 1].imag])

    return line_time, point_time, line_amp, line_complex, point_complex, line_radius


anim1 = FuncAnimation(fig1, animate1, init_func=init1, frames=esantioane // 10, interval=40, blit=True)
#plt.show()

# b.

omega = [1, 2, 5, 7]
Y = [x * np.exp(-2j * np.pi * i * t) for i in omega]
dft = [np.mean(y) for y in Y]

fig2, axes = plt.subplots(1, 4, figsize=(15, 6))

desene = []
linii_dft = []
markeri = []
elemente = []

for i in range(4):
    ax = axes[i]
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginar")
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    desen, = ax.plot([], [], color='purple', linewidth=1.5)
    desene.append(desen)
    elemente.append(desen)

    linie, = ax.plot([], [], 'k-', linewidth=1.5)
    linii_dft.append(linie)
    elemente.append(linie)

    marker, = ax.plot([], [], 'ro', markersize=8)
    markeri.append(marker)
    elemente.append(marker)


def init2():
    for d, l, m in zip(desene, linii_dft, markeri):
        d.set_data([], [])
        l.set_data([], [])
        m.set_data([], [])
    return elemente


def animate2(i):
    step = 10
    end_index = i * step

    if end_index > esantioane:
        end_index = esantioane

    for j in range(4):
        desene[j].set_data(Y[j][:end_index].real, Y[j][:end_index].imag)
        linii_dft[j].set_data([0, Y[j][end_index].real], [0, Y[j][end_index].imag])

        if end_index > 0:
            markeri[j].set_data(Y[j][end_index - 1].real, Y[j][end_index - 1].imag)
            linii_dft[j].set_data([0, Y[j][end_index - 1].real], [0, Y[j][end_index - 1].imag])

    return elemente

anim2 = FuncAnimation(fig2, animate2, init_func=init2, frames=esantioane // 10, interval=40, blit=False)
plt.tight_layout()
#plt.show()

with PdfPages("PDF.pdf") as pdf:
    pdf.savefig(fig1)
    pdf.savefig(fig2)



#3.
frecventaEsantionare = 1000
durata = 1
esantioane = int(frecventaEsantionare * durata)

t = np.arange(0, durata, 1 / frecventaEsantionare)

f1, A1 = 10, 1
f2, A2 = 20, 2
f3, A3 = 75, 0.5

x = (A1 * np.sin(2 * np.pi * f1 * t) +
     A2 * np.sin(2 * np.pi * f2 * t) +
     A3 * np.sin(2 * np.pi * f3 * t))

X = np.fft.fft(x)
modul = np.abs(X)
frecventa2 = np.fft.fftfreq(esantioane, 1 / frecventaEsantionare)

jumate = esantioane // 2
frecventaDesen = frecventa2[:jumate]
modulDesen = modul[:jumate]

fig3, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(t, x, label='$x(t)$')
axes[0].set_xlabel('Timp (s)')
axes[0].set_ylabel('$x(t)$')
axes[0].set_title('Semnal in domeniul timp')
axes[0].grid(True)
axes[0].set_xlim(0, durata)
axes[0].set_ylim(-4, 4)

axes[1].stem(frecventaDesen, modulDesen, basefmt=" ", markerfmt="o", linefmt='-', label='$|x(\omega)|$')
axes[1].set_xlabel('Frecventa (Hz)')
axes[1].set_ylabel('$|x(\omega)|$')
axes[1].set_title('Modulul Transformatei Fourier')
axes[1].set_xlim(0, 100)
axes[1].set_ylim(0, 1200)
axes[1].grid(True)
axes[1].ticklabel_format(style='plain', axis='y')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print(f"\nFrecventele introduse: {f1} Hz, {f2} Hz, {f3} Hz")

peak_indices = np.argsort(modulDesen)[-3:][::-1]
print(f"Frecventele identificate (top 3): {frecventaDesen[peak_indices]} Hz")

with PdfPages("PDF2.pdf") as pdf:
    pdf.savefig(fig3)

