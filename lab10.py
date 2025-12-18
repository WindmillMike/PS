import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso

#1.
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

#2.
predictii = 100
stop = N + predictii # nu ma lasa sa pun cu N + predictii in for, nu am idee de ce
p = 50
Y = serieTimp[p:]
X = np.zeros((len(Y), p + 1))
serie_prezisa = np.concatenate((serieTimp, np.full(predictii, np.nan)))
t_predictii = np.arange(N + predictii)

for i in range(len(Y)):
    X[i, 1:] = serieTimp[i : p + i][::-1]  # Se creaza matricea din slide-ul 15 al cursului (cea cu y fara caciula) doar ca noi avem seria de timp si doar incercam sa o prezicem si deci o construim pe toata de la p pana la 1000
    X[i, 0] = 1

coeficienti_ar = np.linalg.lstsq(X, Y, rcond=None)[0]  # calculam x din slide-ul 15 al cursului

c = coeficienti_ar[0]
phi = coeficienti_ar[1:]

for i in range(N, stop):
    valori_anterioare = serie_prezisa[i - p: i][::-1]
    predictia_t = c + np.dot(phi, valori_anterioare)
    serie_prezisa[i] = predictia_t

plt.figure(figsize=(15, 6))
plt.plot(t_predictii[:N], serieTimp, color='C0')
plt.plot(t_predictii[N - 1:], serie_prezisa[N - 1:], color='red', linestyle='-', linewidth=2)
plt.axvline(x = N, color='gray', linestyle=':')
plt.grid(True)
#plt.show()

#3.
#Greedy (e foarte incet,dar merge mai bine decat 2)
# parametrii_selectie = 600
# predictii = 100
# stop = N + predictii
# cei_mai_buni_parametrii = []
# eroare_minima = np.inf
# eroare_curenta = np.inf
# parametrii = 50
#
# Y = serieTimp[parametrii_selectie:]
# X = np.zeros((len(Y), parametrii_selectie + 1))
#
# for i in range(len(Y)):
#     X[i, 1:] = serieTimp[i : parametrii_selectie + i][::-1]  # Se creaza matricea din slide-ul 15 al cursului (cea cu y fara caciula) doar ca noi avem seria de timp si doar incercam sa o prezicem si deci o construim pe toata de la p pana la 1000
#     X[i, 0] = 1
#
# for i in range(parametrii):
#     eroare_parametru = {}
#     parametrii_ramasi = [j for j in range(parametrii_selectie) if j not in cei_mai_buni_parametrii]
#     for parametru in parametrii_ramasi:
#         parametrii_netestati = cei_mai_buni_parametrii + [parametru]   # Vrem sa testam daca parametrul pe care il adaugam ne va scadea eroarea sau nu
#         X_parametru = np.hstack([np.ones((len(Y), 1)), X[:, parametrii_netestati]])
#         coeficienti_ar = np.linalg.lstsq(X_parametru, Y, rcond=None)[0]
#         predctia = X_parametru @ coeficienti_ar
#         mse = mean_squared_error(Y, predctia)
#
#         eroare_parametru[parametru] = mse
#     if not eroare_parametru:
#         break
#     parametrul_candidat = min(eroare_parametru, key = eroare_parametru.get)
#
#     if eroare_parametru[parametrul_candidat] < eroare_curenta:
#         cei_mai_buni_parametrii.append(parametrul_candidat)
#         eroare_curenta = eroare_parametru[parametrul_candidat]
#     else:
#         break
#
# model_final = sorted(cei_mai_buni_parametrii)
# X_final = np.hstack([np.ones((len(Y), 1)), X[:, model_final]])
# coef_greedy = np.linalg.lstsq(X_final, Y, rcond=None)[0]
# c_greedy = coef_greedy[0]
# phi_greedy = coef_greedy[1:]
# phi_greedy1 = np.zeros(parametrii_selectie)
# for i, indice_lag in enumerate(model_final):
#     phi_greedy1[indice_lag] = phi_greedy[i]
# print(phi_greedy1)
#
# serie_prezisa = np.concatenate((serieTimp, np.full(predictii, np.nan)))
#
# for i in range(N, stop):
#     fereastra_trecut = serie_prezisa[i - parametrii_selectie : i][::-1]
#     serie_prezisa[i] = c_greedy + np.dot(phi_greedy1, fereastra_trecut)
#
# plt.figure(figsize=(15, 6))
# plt.plot(t_predictii[:N], serieTimp, color='C0')
# plt.plot(t_predictii[N - 1:], serie_prezisa[N - 1:], color='red', linestyle='-', linewidth=2)
# plt.axvline(x=N, color='gray', linestyle=':')
# plt.grid(True)
# plt.show()

#L1
# parametrii_L1 = 600
# Y_L1 = serieTimp[parametrii_L1:]
# X_L1 = np.zeros((len(Y_L1), parametrii_L1))
#
# for i in range(len(Y_L1)):
#     X_L1[i, :] = serieTimp[i:parametrii_L1+i][::-1]
#
# lambda_lasso = 0.1
# model_L1 = Lasso(alpha = lambda_lasso, fit_intercept = True, max_iter = 10000)
# model_L1.fit(X_L1, Y_L1)
#
# phi_L1 = model_L1.coef_
# c_L1 = model_L1.intercept_
#
# parametrii = np.where(np.abs(phi_L1) > 1e-5)[0]
# parametrii = sorted(parametrii)
#
# predictii = 100
# stop = N + predictii
# serie_prezisa_lasso = np.concatenate((serieTimp, np.full(predictii, np.nan)))
#
# for i in range(N, stop):
#     valori_trecut = [serie_prezisa_lasso[i - (parametru + 1)] for parametru in parametrii]
#     phi2 = phi_L1[parametrii]
#     serie_prezisa_lasso[i] = c_L1 + np.dot(phi2, valori_trecut)
#
# plt.figure(figsize=(15, 6))
# plt.plot(range(N), serieTimp)
# plt.plot(range(N, stop), serie_prezisa_lasso[N:], color='red')
# plt.axvline(x=N, color='gray', linestyle=':')
# plt.show()

#4.
#coeficientii sa fie de forma x^2 + 3x + 5     coef = [1, 3, 5]
def matrice_companion(coeficienti):
    coeficienti = np.array(coeficienti, dtype=float)
    if coeficienti[0] != 1:
        coeficienti /= coeficienti[0]
    n = len(coeficienti) - 1
    matrice_companion = np.zeros((n, n))
    for i in range(n - 1):
        matrice_companion[i + 1, i] = 1
    for i in range(n):
        matrice_companion[i, -1] -= coeficienti[::-1][i]

    print(matrice_companion)
    return np.linalg.eigvals(matrice_companion)


coef = [1, -3, 2, 4]
radacini = matrice_companion(coef)

#5.


# Lag = matrice_companion(coef_greedy[::-1])
# print(np.all(np.abs(Lag) > 1 + 1e-10))

# L1 = np.append(phi_L1[::-1], c_L1)
# Lag = matrice_companion(L1)
# print(np.all(np.abs(Lag) > 1 + 1e-10))

