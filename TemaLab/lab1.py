import numpy as np
import matplotlib.pyplot as plt

#1.


def func(frecventa):
    t = np.arange(0, 0.03, 1 / frecventa)
    x = np.cos(520 * np.pi * t + np.pi / 3)
    y = np.cos(280 * np.pi * t - np.pi / 3)
    z = np.cos(120 * np.pi * t + np.pi / 3)
    return x, y, z, t


# a
x, y, z, t = func(2000)

plt.figure(figsize=(15, 10))


# b
def figura(x, t, ylabel, nr):
    plt.subplot(3, 1, nr)
    plt.plot(t, x)
    plt.xlabel('timp')
    plt.ylabel(ylabel)


def proiectareFig(x, y, z, t):
    figura(x, t, 'x(t)', 1)
    figura(y, t, 'y(t)', 2)
    figura(z, t, 'z(t)', 3)


proiectareFig(x, y, z, t)

plt.tight_layout()
#plt.show()

# c.

x, y, z, t = func(200)


def figura(x, t, ylabel, nr):
    plt.subplot(3, 1, nr)
    plt.stem(t, x)
    plt.xlabel('timp')
    plt.ylabel(ylabel)


plt.figure(figsize=(15, 10))

proiectareFig(x, y, z, t)

plt.tight_layout()
#plt.show()

#2.
#a.
frecventa = 400
nrEsantioane = 1600
rataEsantionare = 16000
t = np.arange(0, 1/10, 1/rataEsantionare)
x = np.sin(2 * np.pi * 400 * t)

plt.figure(figsize=(15, 10))
plt.plot(t, x)
plt.xlabel('timp')
plt.ylabel('Amplitudine')
plt.grid(True)
plt.show()


#b.
frecventa = 800
durata = 3
rataEsantionare = 16000
t = np.arange(0, 3, 1/rataEsantionare)
x = np.sin(2 * np.pi * 800 * t)

plt.figure(figsize=(15, 10))
plt.plot(t, x)
plt.xlabel('timp')
plt.ylabel('Amplitudine')
plt.xlim(0, 0.005)
plt.grid(True)
#plt.show()


#c.
frecventa = 240
rataEsantionare = 8000
t = np.arange(0, 5/frecventa, 1/rataEsantionare)
x = 2 * (t * frecventa - np.floor(0.5 + t * frecventa))

plt.figure(figsize=(15, 10))
plt.plot(t, x)
plt.xlabel('timp')
plt.ylabel('Amplitudine')
plt.grid(True)
#plt.show()


#d.
frecventa = 300
rataEsantionare = 8000
t = np.arange(0, 5/frecventa, 1/rataEsantionare)
x = np.sign(np.sin(2 * np.pi * frecventa * t))

plt.figure(figsize=(15, 10))
plt.plot(t, x)
plt.xlabel('timp')
plt.ylabel('Amplitudine')
plt.grid(True)
#plt.show()


#e.
I = np.random.rand(128, 128)

plt.figure(figsize=(15, 10))
plt.imshow(I)
#plt.show()


#f.
I2 = np.zeros((128, 128))

cx, cy = 64, 64
for i in range(128):
    for j in range(128):
        if (i - cx)**2 + (j - cy)**2 < 30**2:
            I2[i, j] = 1


plt.figure(figsize=(15, 10))
plt.imshow(I2)
#plt.show()


#3.
#a. 1/frecventa
#b.1 ora = 60 minute, 60 minute = 3600 sec, 3600 sec = 3.600.000ms
#1/2000 = 0.5 ms. Deci 1 esantion e 0.5 ms si 4 biti
#3.600.000 / 0.5 = 7.200.000 * 4 = 28.800.000 / 8 = 3.600.000 bytes
