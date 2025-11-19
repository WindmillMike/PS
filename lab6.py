import numpy as np
import matplotlib.pyplot as plt

#1.
durata = 3
t = np.arange(-durata, durata, 1/500)
x = np.sinc(t) ** 2
frecvente = [1, 1.5, 2, 4]

def calcEsantioane(durata, frecventa, t):
    Ts = 1 / frecventa
    durataEsantioane = np.ceil(durata * frecventa)
    timp = np.arange(-durataEsantioane, durataEsantioane)
    timpEsantioane = timp * Ts
    xEsantioane = np.sinc(timpEsantioane) ** 2

    t = t.reshape(-1, 1)
    sincTimp = (t - timpEsantioane) / Ts  #asta e partea de (t-nTs)/Ts doar ca timpEsantioane e deja timp * Ts
    xPunctat = np.sinc(sincTimp) * xEsantioane
    xPunctat = np.sum(xPunctat, axis=1)

    return xEsantioane, timpEsantioane, xPunctat

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for i in range(2):
    for j in range(2):
        xEsantioane, timpEsantioane, xPunctat = calcEsantioane(durata, frecvente[2 * i + j], t)
        axes[i][j].plot(t, x, color='k', linewidth=2)
        axes[i][j].plot(t, xPunctat, color='#2ca02c', linestyle='--', linewidth=2)
        axes[i][j].stem(timpEsantioane, xEsantioane, linefmt='C1-', markerfmt='C1o', basefmt='k')
        axes[i][j].set_xlim([-3, 3])
        axes[i][j].set_ylim([-0.2, 1.1])
        axes[i][j].grid(True, linestyle=':', alpha=0.5)
        axes[i][j].axhline(0, color='black', linewidth=2)

plt.tight_layout()
#plt.show()

#cu cat marim B mai mult tinde catre Dirac

#2.
x = np.random.rand(100)
x2 = x ** 2
x3 = x ** 3
x4 = x ** 4

fig, axes = plt.subplots(4, 1, figsize=(15, 10))
axes[0].plot(x)
axes[1].plot(x2)
axes[2].plot(x3)
axes[3].plot(x4)

plt.tight_layout()
#plt.show()

frecventa = 1
t = np.arange(-0.5, 1, 1 / 200)
x = np.sign(np.sin(frecventa * t * np.pi * 2))
x2 = x ** 2
x3 = x ** 3
x4 = x ** 4

fig, axes = plt.subplots(4, 1, figsize=(15, 10))
axes[0].plot(x)
axes[1].plot(x2)
axes[2].plot(x3)
axes[3].plot(x4)

plt.tight_layout()
#plt.show()

#3.
n = 10
size = np.random.randint(3, n)
p1 = np.random.randint(1, n, size = size)
p2 = np.random.randint(1, n, size = size)
p3 = np.convolve(p1, p2)
#print(p1, p2)
#print(p3)

lungime = len(p1) + len(p2) - 1
P = np.fft.fft(p1, n = lungime)
Q = np.fft.fft(p2, n = lungime)

R = P * Q
r = np.fft.ifft(R)
r = np.real_if_close(r)
r = np.round(r).astype(int)
#print(r)

#4.
n = 20
t = np.arange(0, 1, 1 / n)
x = np.sin(2 * np.pi * t)
d = np.random.randint(n)
#print(d)
x = x + np.random.rand(n)
y = np.roll(x, d)

P = np.fft.fft(x)
Q = np.fft.fft(y)
R = P * np.conj(Q)
r = np.fft.ifft(R)

r = np.real(r)
deplasare = np.argmax(r)
deplasare = (n - deplasare) % n

#print(deplasare)

#Conform teoremei de deplasare circulara trebuie sa facem corelatia,nu convolutia pentru a gasi varful ce indica deplasarea
#Daca ar fi sa luam P / Q* ,unde Q* e conjugata lui Q, e o deconvolutie si nu ar da bine.
#Pe de alta parte daca luam P / Q simplu este "decorelatia" si ne da un varf care este deplasarea

#5.

def Dreptunghi(Nw):
    return [1.0] * Nw

def Hanning(Nw):
    lista = []
    for i in range(Nw):
        valoare = 0.5 * (1 - np.cos((2 * np.pi * i) / (Nw - 1)))
        lista.append(valoare)
    return np.array(lista)

frecventaEsantionare = 5000
frecventa = 100
Nw = 200

t = np.arange(0, 5, 1 / frecventaEsantionare)
x = np.sin(2 * np.pi * frecventa * t)

fereastra1 = Dreptunghi(Nw)
fereastra2 = Hanning(Nw)

sin1 = x[1000 : 1000 + Nw] * fereastra1
sin2 = x[1000 : 1000 + Nw] * fereastra2

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t[1000 : 1000 + Nw], sin1, label='Spectru', color='C0')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t[1000 : 1000 + Nw], sin2, label='Spectru', color='C1')
plt.grid(True)

plt.tight_layout()
#plt.show()

#6.
#a.
x = np.genfromtxt("Train.csv", delimiter=",", skip_header=1)
x2 = x[:72, 2]

#b.
w = [5, 10, 15, 20]
for i in w:
    x = np.convolve(x2, np.ones(i), 'valid') / i
    print(x)
