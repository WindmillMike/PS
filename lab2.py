import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io.wavfile import write, read

# 1.
amplitudine = 2.0
frecventa = 5
rata_esantionare = 100
durata = 2
deplasare_faza = -np.pi / 2
puncte_totale = int(rata_esantionare * durata)

t = np.linspace(0, durata, puncte_totale, endpoint=False)

sin = amplitudine * np.sin(2 * np.pi * frecventa * t)

cos = amplitudine * np.cos((2 * np.pi * frecventa * t) + deplasare_faza)

plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
plt.plot(t, sin)
plt.xlabel('timp')
plt.ylabel('amplitudine')
plt.grid(True)
plt.ylim(-1.1 * amplitudine, 1.1 * amplitudine)

plt.subplot(2, 1, 2)
plt.plot(t, cos)
plt.xlabel('timp')
plt.ylabel('amplitudine')
plt.grid(True)
plt.ylim(-1.1 * amplitudine, 1.1 * amplitudine)

plt.tight_layout()
# plt.show()

# 2.
amplitudine = 2.0
frecventa = 3
rata_esantionare = 100
durata = 2
puncte_totale = int(rata_esantionare * durata)
t = np.linspace(0, durata, puncte_totale, endpoint=False)
faze = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

plt.figure(figsize=(15, 10))
plt.xlabel('timp')
plt.ylabel('amplitudine')
plt.grid(True)
plt.ylim(-1.1 * amplitudine, 1.1 * amplitudine)

for faza in faze:
    semnal = amplitudine * np.sin(2 * np.pi * frecventa * t + faza)
    plt.plot(t, semnal)

# plt.show()

x = amplitudine * np.sin(2 * np.pi * frecventa * t)
z = np.random.normal(size=puncte_totale)

snr_dorite = [0.1, 1, 10, 100]

norma_x_patrat = np.linalg.norm(x) ** 2
norma_z_patrat = np.linalg.norm(z) ** 2

plt.figure(figsize=(15, 10))

for i, snr in enumerate(snr_dorite):
    gamma = np.sqrt(norma_x_patrat / (snr * norma_z_patrat))

    x_zgomot = x + gamma * z

    plt.subplot(len(snr_dorite), 1, i + 1)
    plt.plot(t, x_zgomot)
    plt.xlabel('timp')
    plt.ylabel('amplitudine')
    plt.grid(True)
    plt.ylim(-2.5, 2.5)

plt.tight_layout()
#plt.show()

# 3.
frecventa = 400
nrEsantioane = 1600
rataEsantionare = 3200
t = np.arange(0, 1, 1 / rataEsantionare)
x = np.sin(2 * np.pi * frecventa * t)

write('semnal.wav', rataEsantionare, x)

fs, data = read('semnal.wav')

# sd.play(data, fs)
sd.wait()

# 4.
frecventa = 2
durata = 1.5
rata_esantionare = 200
amplitudine = 1.0
nrEsantioane = rata_esantionare * durata
t = np.arange(0, durata, 1 / rata_esantionare)

sinus = amplitudine * np.sin(2 * np.pi * frecventa * t)

sawtooth_baza = np.fmod(frecventa * t, 1)
sawtooth = 2 * amplitudine * (sawtooth_baza - 0.5)

suma_semnale = sinus + sawtooth

plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(t, sinus)
plt.xlabel('timp')
plt.ylabel('amplitudine')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, sawtooth)
plt.xlabel('timp')
plt.ylabel('amplitudine')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, suma_semnale)
plt.xlabel('timp')
plt.ylabel('amplitudine')
plt.grid(True)

plt.tight_layout()
#plt.show()

# 5.
rata_esantionare_comun = 44100

frecventa1 = 200
durata1 = 1.5
amplitudine1 = 1.0
t1 = np.arange(0, durata1, 1 / rata_esantionare_comun)

sin1 = amplitudine * np.sin(2 * np.pi * frecventa1 * t1)

frecventa2 = 400
durata2 = 1
amplitudine2 = 1.0
t2 = np.arange(0, durata2, 1 / rata_esantionare_comun)

sin2 = amplitudine * np.sin(2 * np.pi * frecventa2 * t2)

sin3 = np.concatenate((sin1, sin2))

# sd.play(sin3, rata_esantionare_comun)
sd.wait()

# 6.
frecventa = 100
amplitudine = 1.0
durata = 1.0
rata_esantionare = 100
t = np.arange(0, durata, 1 / rata_esantionare)
# a.
f_a = frecventa / 2
x_a = amplitudine * np.sin(2 * np.pi * f_a * t)

# b.
f_b = frecventa / 4
x_b = amplitudine * np.sin(2 * np.pi * f_b * t)

# c.
f_c = 0
x_c = amplitudine * np.sin(2 * np.pi * f_c * t)

plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(t, x_a)
plt.xlabel('timp')
plt.ylabel('amplitudine')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, x_b)
plt.xlabel('timp')
plt.ylabel('amplitudine')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, x_c)
plt.xlabel('timp')
plt.ylabel('amplitudine')
plt.grid(True)

plt.tight_layout()
# plt.show()

# 7.
frecventa = 100
durata = 0.2
rata_esantionare = 1000
t = np.arange(0, durata, 1 / rata_esantionare)

x = np.sin(2 * np.pi * frecventa * t)

t2 = t[::4]
x2 = x[::4]

t3 = t[1::4]
x3 = x[1::4]

plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.xlabel('timp')
plt.ylabel('amplitudine')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t2, x2)
plt.xlabel('timp')
plt.ylabel('amplitudine')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t3, x3)
plt.xlabel('timp')
plt.ylabel('amplitudine')
plt.grid(True)

plt.tight_layout()


# plt.show()

# 8.
def approx_pade(a):
    return (a - 7 * a ** 3 / 60) / (1 + a ** 2 / 20)


alpha = np.linspace(-np.pi / 2, np.pi / 2, 500)

y_real = np.sin(alpha)
y_taylor = alpha
y_pade = approx_pade(alpha)

eroare_taylor = y_taylor - y_real
eroare_pade = y_pade - y_real


plt.figure(figsize=(15, 10))
plt.plot(alpha, y_real)
plt.plot(alpha, y_taylor)
plt.plot(alpha, y_pade)
plt.xlabel(r'$\alpha$ (radiani)')
plt.ylabel('valoare')
plt.grid(True)

plt.figure(figsize=(15, 10))
plt.plot(alpha, eroare_taylor)
plt.plot(alpha, eroare_pade)
plt.xlabel(r'$\alpha$ (radiani)')
plt.ylabel('Eroare (liniară)')
plt.grid(True)

plt.figure(figsize=(15, 10))
plt.plot(alpha, np.abs(eroare_taylor))
plt.plot(alpha, np.abs(eroare_pade))
plt.yscale('log')
plt.xlabel(r'$\alpha$ (radiani)')
plt.ylabel('Eroare Absolută (log)')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()
