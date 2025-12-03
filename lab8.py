import numpy as np
import matplotlib.pyplot as plt
#a.
N = 1000
t = np.arange(N)
trend = 1 / 10000 * t ** 2 + 1 / 1000 * t + 50

ampl, frecventa = 2, 1 / 50
ampl2, frecventa2 = 3, 1 / 20
sezon = ampl * np.sin(2 * np.pi * frecventa * t) + ampl2 * np.cos(2 * np.pi * frecventa2 * t)

zgomot = np.random.normal(0, 1, N)

serieTimp = trend + sezon + zgomot

plt.figure(figsize=(15, 10))

plt.subplot(4, 1, 1)
plt.plot(t, serieTimp, color='C0')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(t, trend, color='C1')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t, sezon, color='C2')
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(t, zgomot, color='C3')
plt.grid(True)

plt.tight_layout()
#plt.show()

#b.
serieFaraDC = serieTimp - np.mean(serieTimp)
covarianta = np.correlate(serieFaraDC, serieFaraDC, mode='full')  # https://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation
covarianta = covarianta[covarianta.size // 2:]
autocorelatie = covarianta / covarianta[0] # transformam rezultatul in coeficientul de corelatie prin impartirea la varianta

plt.figure(figsize=(10, 5))
plt.stem(t[:200 + 1], autocorelatie[:200 + 1], linefmt='C0-', markerfmt='C0o', basefmt='k-')

plt.ylim(-1.0, 1.5)
plt.grid(axis='y', linestyle='--')
#plt.show()

#c.
p = 2
Y = serieTimp[p:]
X = np.zeros((len(Y), p + 1))

for i in range(len(Y)):
    X[i, 1:] = serieTimp[i : p + i][::-1]  #Se creaza matricea din slide-ul 15 al cursului (cea cu y fara caciula) doar ca noi avem seria de timp si doar incercam sa o prezicem si deci o construim pe toata de la p pana la 1000
    X[i, 0] = 1

coeficienti_ar = np.linalg.lstsq(X, Y, rcond=None)[0]  # calculam x din slide-ul 15 al cursului

c = coeficienti_ar[0]
phi = coeficienti_ar[1:]
predictii = np.full(N, np.nan)

for i in range(p, N):
    valori_anterioare = serieTimp[i - p : i][::-1]
    predictia_t = c + np.dot(phi, valori_anterioare)
    predictii[i] = predictia_t

plt.figure(figsize=(14, 6))
plt.plot(t[:500], serieTimp[:500], color='C0')
plt.plot(t[:500], predictii[:500], color='red', linestyle='--', linewidth=2)
plt.axvline(x=p, color='gray', linestyle=':')

plt.legend()
plt.grid(True)
#plt.show()

