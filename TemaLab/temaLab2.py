import numpy as np
import matplotlib.pyplot as plt
from scipy.datasets import ascent # nu imi gasea modulul misc
from scipy.fft import dctn, idctn
from Q_jpeg import Q_jpeg, adaugarePadding
from functiiUtilitare import spirala, calcul_size, huffman_codare, calculare_bitstream, transformare_octeti
import Flag as f

np.set_printoptions(suppress=True,
                    formatter={'float_kind':'{:f}'.format})

X = ascent()
inaltime, latime = X.shape

Q_jpeg = Q_jpeg  #matricea de cuantizare

X_padding = adaugarePadding(X)
inaltime_padding, latime_padding = X_padding.shape

dc_anterior = 0
huffman = [] # aici vor intra valorile RLE pentru huffman
reconstruit = np.zeros_like(X_padding, dtype=float)
for i in range(inaltime_padding // 8):
    for j in range(latime_padding // 8):
        bloc = X_padding[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
        bloc_centrat = bloc - 128.0
        bloc_dct = dctn(bloc_centrat).astype(float)
        bloc_dct_impartit = np.round(bloc_dct / Q_jpeg).astype(np.int32)
        #print(np.max(np.abs(bloc_dct_impartit)))

        bloc_spirala = spirala(bloc_dct_impartit) # aici facem zig-zagul pentru ca 0-urile sa fie grupate frumos

        dc_actual = bloc_spirala[0] - dc_anterior  # se realizeaza pentru ca diferenta de luminozitate intre blocuri sa fie mai mica si sa nu iimi dea valori care nu sunt in tabel
        dc_anterior = bloc_spirala[0]

        size_dc = calcul_size(dc_actual)
        ac = bloc_spirala[1:]
        huffman_codare(ac, huffman, size_dc, dc_actual)

        bloc_inmultit = Q_jpeg * bloc_dct_impartit
        bloc_convertit_inapoi = idctn(bloc_inmultit)
        reconstruit[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = bloc_convertit_inapoi

reconstruit += 128.0
reconstruit = np.clip(reconstruit, 0, 255).astype(np.uint8)

bitstream = calculare_bitstream(huffman)
bitstream_octeti = transformare_octeti(bitstream)

print(Q_jpeg)
print(f.construieste_dqt(Q_jpeg))
print(f.construieste_sof0(inaltime, latime))

# plt.subplot(2, 1, 1)
# plt.imshow(X, cmap='gray')
# plt.subplot(2, 2, 1)
# plt.imshow(reconstruit, cmap='gray')
# plt.axis('off')
# plt.show()

