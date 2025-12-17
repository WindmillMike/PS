import numpy as np
import matplotlib.pyplot as plt

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
p = 600
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
plt.show()

#3.
#Greedy
best_phi = []
best_error = np.inf
eroare_curenta = np.inf
phi_ramase = 100

for k in range(phi_ramase):
    mse_with_new_lag = {}
    phi_ramase = [l for l in range(p) if l not in best_phi]

    for i in phi_ramase:
        # Lag-ul dorit (Lag_idx=0 înseamnă Y_t-1, Lag_idx=1 înseamnă Y_t-2, etc.)
        lags_to_test = best_phi + [i]

        # Construim matricea de regresie X_k (adaugam coloana de 1 pentru intercept)
        X_k = np.hstack([np.ones((len(Y), 1)), X[:, lags_to_test]])

        # Antrenăm modelul (OLS)
        w_k = np.linalg.lstsq(X_k, Y, rcond=None)[0]

        # Calculăm MSE pe setul de antrenare (o metrică proxy pentru calitatea modelului)
        y_pred_k = X_k @ w_k
        mse = mean_squared_error(Y_greedy, y_pred_k)

        mse_with_new_lag[i] = mse

    # Găsim lag-ul care a minimizat MSE-ul
    if not mse_with_new_lag: break
    best_candidate_lag = min(mse_with_new_lag, key=mse_with_new_lag.get)

    # Dacă noul MSE este mai bun, adăugăm lag-ul
    if mse_with_new_lag[best_candidate_lag] < current_mse:
        best_phi.append(best_candidate_lag)
        current_mse = mse_with_new_lag[best_candidate_lag]
    else:
        break  # Ne-am oprit din îmbunătățire

# 3. CONSTRUIREA MODELULUI FINAL GREEDY SPARSE
# Deoarece indicii încep de la 0, le adăugăm +1 pentru a avea lag-urile AR (Y_t-1, Y_t-2...)
lags_final = [l + 1 for l in sorted(best_phi)]
print(f"\nLag-uri selectate prin metoda Greedy: {lags_final}")

# Re-antrenăm modelul final AR(sparse)
p_greedy = len(lags_final)
X_final_greedy = np.hstack([np.ones((len(Y_greedy), 1)), X_greedy[:, best_phi]])
coef_greedy = np.linalg.lstsq(X_final_greedy, Y_greedy, rcond=None)[0]
c_greedy = coef_greedy[0]
phi_greedy = coef_greedy[1:]
print(f"Număr de coeficienți non-zero (excluzând c): {p_greedy}")
