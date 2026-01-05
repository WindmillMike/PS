import numpy as np

Q_jpeg = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 28, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109,103, 77],
    [24, 35, 55, 64, 81, 104,113, 92],
    [49, 64, 78, 87, 103,121,120,101],
    [72, 92, 95, 98, 112,100,103, 99]
], dtype=float)

f = 2
Q_jpeg = np.round(Q_jpeg * f).astype(np.int32)

def adaugarePadding(X2):
    inaltime, latime = X2.shape
    n, m = (-latime) % 8, (-inaltime) % 8
    ceva = np.zeros((inaltime + m, latime + n), dtype=X2.dtype)
    ceva[:inaltime, :latime] = X2
    return ceva

