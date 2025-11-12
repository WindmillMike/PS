import numpy as np
import matplotlib.pyplot as plt

#1.
#a.
#1 esantion / ora
#b.
#2 ani si 32 de zile
#c.
N = 18288
x = np.genfromtxt("Train.csv", delimiter=",", skip_header=1)
x2 = x[:, 2]
f_max = np.max(x2, axis = 0)
#print(f_max)

#d.
frecventaEsantionare = 1/3600

x3 = np.fft.fft(x2) / N
modul = np.abs(x3)
modul = modul[:N // 2]

frecventa = frecventaEsantionare * np.linspace(0, N/2, N//2, endpoint=False) / N

plt.figure(figsize=(10, 5))
plt.plot(frecventa, modul)
plt.grid(True)
#plt.show()

#e.Folosim mean din numpy ca sa calculam media semnalului
media = np.mean(x2) # sau x3[0]
#Stim ca exista o componenta continua asa ca incercam sa o eliminam
x_medie = x2 - media

#f.
x3 = np.fft.fft(x_medie) / N
modul1 = np.abs(x3)
modul1 = modul1[1:N // 2]
indici_sortati = np.argsort(modul1)[::-1]
indici_sortati = indici_sortati[:4] + 1
for i in indici_sortati:
    frecventaCicluri = frecventa[i]
    # print(frecventaCicluri)

#Caror perioade corespunde?
#Stim ca timpul e 1/frecventa si deci
#1/1.518907358802372e-08 ne da aproximativ 2 ani    nu am nici o idee
#1/3.037814717604744e-08 ne da aproximativ 1 an     anul nou
#1/1.1574074074074073e-05 ne da o zi    probabil diferenta zi/noapte
#1/4.5567220764071157e-08 ne da 8       schimbari de anotimp

#g
#esantionul e unul pe ora deci o zi sunt 24 de esantionae, saptamana e 7 * 24 = 168
#deci ne trebuie multiplu de 168 > 1000 asa ca il luam pe primul care e 6 si ne da
#8 * 168 = 1440 esantioane (evident ca am plecat de la presupunerea ca prima zi din Train.csv e o luni ora 00:00)

trafic = x2[1440:2160]
timp = np.arange(0, 720)

plt.figure(figsize=(15, 6))
plt.plot(timp, trafic)
plt.xlabel('Timpul (Ore, unde 0 = Eșantionul 1440)')
plt.ylabel('Volumul de Mașini')
plt.grid(True)
plt.tight_layout()
plt.show()

#h.
#Putem selecta mai multe saptamani si sa analizam zilele cu media cea mai mica si sa ne asumam ca sunt sambata
#si duminica.Dupa ce gasim cateva saptamani sa se potriveasca avem offsetul si putem sa ne asumam
#care e prima zi de la care s-a inceput masuratoarea
#Problema este ca duminica seara lumea din provincie pleaca din Bucuresti sau vine si e destul de
#imprevizibil fluxul de persoane + de sarbatori traficul e mai mare si poate nimeri in weekend ceea ce deruteaza potrivirea saptamanilor, etc.)
#Solutia e foarte dependenta de faptul ca oamenii au o rutina previzibila de Luni-Vineri munca
#vineri seara pleaca inapoi in provincie cine are nevoie si duminica se intorc, fara sarbatori sau
#alti factori care sa intervina cu minimele