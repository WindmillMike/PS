import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from moviepy.editor import AudioFileClip
from scipy.io import wavfile
import os

#2.
frecventaNYQ = 300
frecventaPlotare = 100000
durata = 0.03
A = 1

pas_esantionare = 1 / frecventaNYQ
pas_plotare = 1 / frecventaPlotare

frecventa1 = 700
frecventa2 = 1600
frecventa3 = 100

t1 = np.arange(0, durata, pas_plotare)
t2 = np.arange(0, durata, pas_esantionare)

x1 = A * np.sin(2 * np.pi * frecventa1 * t1)
x2 = A * np.sin(2 * np.pi * frecventa2 * t1)
x3 = A * np.sin(2 * np.pi * frecventa3 * t1)

x_samp = A * np.sin(2 * np.pi * frecventa1 * t2)

fig, axes = plt.subplots(4, 1, figsize=(8, 8))

axes[0].plot(t1, x1, color='mediumpurple')

axes[1].plot(t1, x1, color='mediumpurple')
axes[1].scatter(t2, x_samp, color='yellow', s=50, zorder=5)

axes[2].plot(t1, x2, color='darkviolet')
axes[2].scatter(t2, x_samp, color='yellow', s=50, zorder=5)

axes[3].plot(t1, x3, color='green')
axes[3].scatter(t2, x_samp, color='yellow', s=50, zorder=5)
axes[3].set_xlabel('Timp (s)')

for ax in axes:
    ax.grid(True)
    ax.set_xlim(0, durata)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xticks(np.arange(0, 0.030, 0.005))

plt.tight_layout()
#plt.show()

#3.
frecventaNYQ = 1000
frecventaPlotare = 100000
durata = 0.03
A = 1

pas_esantionare = 1 / frecventaNYQ
pas_plotare = 1 / frecventaPlotare

frecventa1 = 700
frecventa2 = 1600
frecventa3 = 100

t1 = np.arange(0, durata, pas_plotare)
t2 = np.arange(0, durata, pas_esantionare)

x1 = A * np.sin(2 * np.pi * frecventa1 * t1)
x2 = A * np.sin(2 * np.pi * frecventa2 * t1)
x3 = A * np.sin(2 * np.pi * frecventa3 * t1)

x_samp = A * np.sin(2 * np.pi * frecventa1 * t2)

fig, axes = plt.subplots(4, 1, figsize=(8, 8))

axes[0].plot(t1, x1, color='mediumpurple')

axes[1].plot(t1, x1, color='mediumpurple')
axes[1].scatter(t2, x_samp, color='yellow', s=50, zorder=5)

axes[2].plot(t1, x2, color='darkviolet')
axes[2].scatter(t2, x_samp, color='yellow', s=50, zorder=5)

axes[3].plot(t1, x3, color='green')
axes[3].scatter(t2, x_samp, color='yellow', s=50, zorder=5)
axes[3].set_xlabel('Timp (s)')

for ax in axes:
    ax.grid(True)
    ax.set_xlim(0, durata)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xticks(np.arange(0, 0.030, 0.005))

plt.tight_layout()
#plt.show()

#4.
#Folosim formula: latimeBanda = 200 - 40 = 160
#frecventaEsantionare = 2 * 200 / m
#m <= 200 / 160
#m = 1
#frecventaEsantionare = 400

#Sau, logic vorbind fiind ca latimea de banda e 160, am putea incerca sa luam doar 320
#dar 160 (intre 40 si 200) aliaza pe 320 la 0.5 si la fel pentru orice sub 400

#5.
clip = AudioFileClip("vocale.mp4")
clip.write_audiofile("vocale_audio.wav", codec='pcm_s16le')
clip.close()

y, sr = librosa.load("vocale_audio.wav", sr=None)

S = np.abs(librosa.stft(y))
S_db = librosa.amplitude_to_db(S, ref=np.max)

plt.figure(figsize=(12, 6))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
plt.colorbar(format="%+2.f dB")
plt.title("Spectrograma vocale (a, e, i, o, u)")
plt.xlabel("Timp (s)")
plt.ylabel("Frecvență (Hz)")
#plt.show()

#Da

#6.
#a.
nume_fisier = 'vocale_audio.wav'

frecventa, data = wavfile.read(nume_fisier)
#a.
if data.ndim > 1:
    semnalAudio = data[:, 0].astype(np.float64)
else:
    semnalAudio = data.astype(np.float64)

if np.max(np.abs(semnalAudio)) != 0:
    semnalAudio /= np.max(np.abs(semnalAudio))

N = len(semnalAudio)
frecventaEsantionare = frecventa
durata = N / frecventaEsantionare
#b.
procent_fereastra = 0.01
dimensiune_fereastra = int(procent_fereastra * N)

suprapunere = 0.50
pas_deplasare = int(dimensiune_fereastra * suprapunere)

matrice_spectrograma = []
fereastra = np.hamming(dimensiune_fereastra)

for start in range(0, N - dimensiune_fereastra + 1, pas_deplasare):
    sfarsit = start + dimensiune_fereastra
    grup = semnalAudio[start:sfarsit]
    grup_fereastra = grup * fereastra
    #c.
    fft_grup = np.fft.fft(grup_fereastra)
    #d.
    fft_grup_mag = np.abs(fft_grup[:pas_deplasare])

    matrice_spectrograma.append(fft_grup_mag)

spectrograma_matrice = np.array(matrice_spectrograma).T
numar_grupuri = spectrograma_matrice.shape[1]


frecvente = np.linspace(0, frecventaEsantionare / 2, spectrograma_matrice.shape[0])
timp = np.linspace(0, durata, numar_grupuri)
amplitudini_dB = 10 * np.log10(spectrograma_matrice / np.max(spectrograma_matrice) + 1e-10)

#e.
plt.figure(figsize=(10, 5))
plt.imshow(amplitudini_dB,
           aspect='auto',
           origin='lower',
           cmap='inferno',
           extent=[timp.min(), timp.max(), frecvente.min(), frecvente.max()])

plt.colorbar(label='Energie (dB)')
plt.xlabel('Timp (s)')
plt.ylabel('Frecvența (Hz)')
plt.title('Spectrograma Vocalelor (STFT Manual)')
plt.show()
