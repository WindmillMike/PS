import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.color import rgb2ycbcr
import scipy.ndimage as ndimage
from scipy.fft import dctn, idctn
from Q_jpeg import Q_jpeg, adaugarePadding
from functiiUtilitare import downsampling, transformareRGB, spirala, calcul_size, huffman_codare, calculare_bitstream, transformare_octeti
import Flag as f

np.set_printoptions(suppress=True,
                    formatter={'float_kind':'{:f}'.format})

X = mpimg.imread("img.png")   # In cazul in care se va citi o imagine diferita de ascent
X = rgb2ycbcr(X)

Y = X[:, :, 0]
Cb = X[:, :, 1]
Cr = X[:, :, 2]

Cb_subsampled = downsampling(Cb)
Cr_subsampled = downsampling(Cr)

Q_jpeg = Q_jpeg  #matricea de cuantizare

def prelucrare(X, parametrulFermecat):
    inaltime, latime = X.shape

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

    bitstream = calculare_bitstream(huffman, parametrulFermecat)
    bitstream_octeti = transformare_octeti(bitstream)

    return reconstruit, bitstream_octeti


Y_prelucrat, bitstream_Y = prelucrare(Y, "LUMINANCE")
Cb_prelucrat, bitstream_Cb = prelucrare(Cb_subsampled, "CHROMINANCE")
Cr_prelucrat, bitstream_Cr = prelucrare(Cr_subsampled, "CHROMINANCE")

Cb_full = ndimage.zoom(Cb_prelucrat, 2, order=1)
Cr_full = ndimage.zoom(Cr_prelucrat, 2, order=1)

Cb_full = Cb_full[:Y_prelucrat.shape[0], :Y_prelucrat.shape[1]]
Cr_full = Cr_full[:Y_prelucrat.shape[0], :Y_prelucrat.shape[1]]

X_reconstruit = np.stack([Y_prelucrat, Cb_full, Cr_full], axis=2)
X_reconstruit = transformareRGB(X_reconstruit)

# print(Q_jpeg)
# print(f.construieste_dqt(Q_jpeg))
# print(f.construieste_sof0(inaltime, latime))


# plt.imshow(X_reconstruit)
# plt.axis('off')
# plt.show()

