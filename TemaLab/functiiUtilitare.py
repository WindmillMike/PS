import numpy as np
from tabele_huffman import HUFFMAN_AC_LUMINANCE, HUFFMAN_DC_LUMINANCE

def spirala(bloc):
    l = 0
    vector = np.zeros(64)
    for i in range(15):
        if i < 8:
            for j in range(i + 1):
                if i % 2 == 0:
                    vector[l] = bloc[i - j][j]
                else:
                    vector[l] = bloc[j][i - j]
                l += 1
        else:
            for j in range(15 - i):
                if i % 2 == 0:
                    vector[l] = bloc[7 - j][i - 7 + j]
                elif i % 2:
                    vector[l] = bloc[i - 7 + j][7 - j]
                l += 1
    return np.array(vector)

def calcul_size(x):
    if x == 0:
        return 1
    return int(np.floor(np.log2(np.abs(x)))) + 1

def huffman_codare(ac, huffman, size_dc, dc_actual):
    huffman.append([])
    huffman[-1].append(("DC", size_dc, dc_actual))
    contor = 0
    for i in range(63):
        valoare_ac = ac[i]
        if valoare_ac == 0:
            contor += 1
            if contor == 16:
                huffman[-1].append(("AC" + str(i), 0xF0))
                contor = 0
        else:
            ac_binar = calcul_size(valoare_ac)
            ac_codat = (contor << 4) | ac_binar
            huffman[-1].append(("AC" + str(i), ac_codat, valoare_ac))
            contor = 0

    if contor > 0:
        huffman[-1].append(("AC EOB", 0x00))

def transformare_in_binar(x, size):
    if x == 0:
        return ""
    x = int(x)
    if x > 0:
        return format(x, f"0{size}b")
    else:
        x2 = (1 << size) - 1 + x
        return format(x2, f"0{size}b")

def calculare_bitstream(huffman):
    bitstream = ""
    for bloc in huffman:
        for pixel in bloc:
            if pixel[0] == "DC":
                size = pixel[1]
                dc_actual = pixel[2]
                size_dc_codat = HUFFMAN_DC_LUMINANCE.get(size)
                dc_binar = transformare_in_binar(dc_actual, size)
                bitstream += size_dc_codat + dc_binar

            elif len(pixel) == 2:
                bitstream += HUFFMAN_AC_LUMINANCE.get(pixel[1])

            elif len(pixel) == 3:
                size_ac_cu_zero = pixel[1]
                valoare = pixel[2]
                size = size_ac_cu_zero & 0x0F
                size_cu_zero_codat = HUFFMAN_AC_LUMINANCE.get(size_ac_cu_zero)
                valoare_binar = transformare_in_binar(valoare, size)
                bitstream += size_cu_zero_codat + valoare_binar

    return bitstream

def transformare_octeti(x):
    padding_len = (8 - (len(x) % 8)) % 8
    x += "0" * padding_len

    v = []
    for i in range(0, len(x), 8):
        octet = x[i:i + 8]
        valoare = int(octet, 2)
        v.append(valoare)

        if valoare == 255:
            v.append(0)

    return v