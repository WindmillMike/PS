from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt


# 1.
# a.

N = 32
n1 = np.arange(N)
n2 = np.arange(N)

n1_grid, n2_grid = np.meshgrid(n1, n2)
x = np.sin(2 * np.pi * n1_grid + 3 * np.pi * n2_grid)  # In cazul asta daca nu scoatem erorile de calcul nu ne mai da rezultatul asteptat
for i, y in enumerate(x):
    for j, u in enumerate(x):
        if x[i][j] < 1e-13:
            x[i][j] = 0

xSpectru = np.fft.fft2(x)
freq_db = 20 * np.log10(abs(xSpectru) + 1e-13) # 1e-13 ca sa nu imparta cu 0

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(x, cmap = "gray")
plt.colorbar(label ='Amplitudine')

plt.subplot(1, 2, 2)
plt.imshow(freq_db, cmap = "gray")
plt.colorbar()

plt.tight_layout()
#plt.show()

# Imaginea nu ne da nimic deoarece in domeniul timp totul e 0 (e sinus si orice numar intreg ne da 0 deci 2 * np.pi * n1_grid + 3 * np.pi * n2_grid e mereu 00
# In spectru ne afiseaza tot uniform deoarece dupa fft de o matrice de 0-uri ne da tot 0

# b.

x = np.sin(4 * np.pi * n1_grid) + np.cos(6 * np.pi * n2_grid)

xSpectru = np.fft.fft2(x)
xSpectru -= xSpectru[0][0]
freq_db = 20 * np.log10(abs(xSpectru) + 1e-16)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(x, cmap = "gray", vmin=0, vmax=1)
plt.colorbar(label ='Amplitudine')

plt.subplot(1, 2, 2)
plt.imshow(freq_db, cmap = "gray")
plt.colorbar()

plt.tight_layout()
# plt.show()

# Imaginea din stanga ar trebui sa ne dea tot uniform ca la a. deoarece sinusul e la fel ca la a si da 0 si cosinusul e multiplu de 2 * pi care e 1 mereu
# Deci semnalul e 1 mereu
# Spectrul e un singur punct negru la stanga sus deoarece acolo se concentreaza toata energia

#c.

Y = np.zeros((N, N), dtype=complex)

Y[0][5] = 1
Y[0][N - 5] = 1

Imagine = np.fft.ifft2(Y)
Imagine = np.real(Imagine)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(Imagine, cmap = "gray")
plt.colorbar(label ='Amplitudine')

plt.subplot(1, 2, 2)
plt.imshow(abs(Y), cmap = "gray")
plt.colorbar()

plt.tight_layout()
#plt.show()

# Imaginea din stanga ne da 5 linii pe verticala construite din frecventa de la Y[0][5] (5 repr. nr. de linii) si Y[0][N - 5] conjugata sa (deoarece e fft si de la jumate sunt doar conjugate)
# punem ca Y[0][N - 5] sa fie si el 1 astfel incat partea complexa a lui Y[0][5] sa se anuleze
# La spectru ne da doar 2 pixeli, energia concentrandu-se doar in ei, Y[0][5] si Y[0][N - 5]

# d.

Y = np.zeros((N, N), dtype = complex)

Y[5][0] = 1
Y[N - 5][0] = 1

Imagine = np.fft.ifft2(Y)
Imagine = np.real(Imagine)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(Imagine, cmap = "gray")
plt.colorbar(label ='Amplitudine')

plt.subplot(1, 2, 2)
plt.imshow(abs(Y), cmap = "gray")
plt.colorbar()

plt.tight_layout()
#plt.show()

# La fel ca la d doar ca liniile sunt pe orizontala

# e.

Y = np.zeros((N, N), dtype = complex)

Y[5][5] = 1
Y[N - 5][N - 5] = 1

Imagine = np.fft.ifft2(Y)
Imagine = np.real(Imagine)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(Imagine, cmap = "gray")
plt.colorbar(label ='Amplitudine')

plt.subplot(1, 2, 2)
plt.imshow(abs(Y), cmap = "gray")
plt.colorbar()

plt.tight_layout()
#plt.show()

# La fel ca anterior doar ca liniile sunt pe diagonala

# 2.

#Nu am avut timp sa il implementez dar as face ce ati prezentat la curs de la JPEG cu matricea Q si Huffman

# 3.
X = misc.face(gray=True)

Y = np.fft.fft2(X)
freq_db = 20 * np.log10(abs(Y))

pixel_noise = 200
noise = np.random.randint(-pixel_noise, high=pixel_noise+1, size=X.shape)
X_noisy = X + noise

signal_power = np.sum(X ** 2)
noise_power = np.sum((X - X_noisy) ** 2)
snr = 10 * np.log10(signal_power / noise_power)
print(snr)

Y = np.fft.fft2(X_noisy)
Yshift = np.fft.fftshift(Y)
cx, cy = len(Yshift) // 2, len(Yshift[0]) // 2
ce = 5
masca = np.zeros_like(Y, dtype=bool)

for i, t in enumerate(Yshift):
    for j, y in enumerate(Yshift[i]):
        if (i - cx) ** 2 + (j - cy) ** 2 >= (cx // ce) ** 2:
            masca[i][j] = False
        else:
            masca[i][j] = True

Yshift *= masca
#filtru = 20 * np.log10(abs(Yshift))
X_cutoff = np.fft.ifft2(np.fft.ifftshift(Yshift))
X_cutoff = np.real(X_cutoff)

signal_power = np.sum(X ** 2)
noise_power = np.sum((X - X_cutoff) ** 2)
snr = 10 * np.log10(signal_power / noise_power)
print(snr)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(X_noisy, cmap = "gray")
plt.colorbar(label ='Amplitudine')

plt.subplot(1, 2, 2)
plt.imshow(X_cutoff, cmap = "gray")
plt.colorbar()
#plt.show()
