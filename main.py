import glob

import numpy as np
from scipy.io import wavfile
from pylab import *
from scipy import *

male_min_max = [55, 158]
female_min_max = [170, 295]
HPSLoop = 6


def harmonic_product_spectrum(sample_rate, data_voice):
    t = 3  # Branie pod uwagę tylko 3 1-sekundowych próbek

    if t > len(data_voice) / sample_rate:
        t = len(data_voice) / sample_rate
    data_voice = data_voice[max(0, int(len(data_voice) / 2) - int(t / 2 * sample_rate)): min(len(data_voice) - 1,
                                                                                             int(len(
                                                                                                 data_voice) / 2) + int(
                                                                                                 t / 2 * sample_rate))]
    # Branie pod uwagę tylko 3 1-sekundowych próbek z analizowanej tablicy
    part_len = int(sample_rate)

    # Podział tablicy na 3 1-sekundowe próbki
    # parts1 = [data_voice[i * part_len: (i + 1) * part_len] for i in range(int(t))]
    parts = [[0 for column in range(part_len)] for row in range(int(t))]
    for i in range(int(t)):
        parts[i] = data_voice[i * part_len: (i + 1) * part_len]
    result_parts = []
    for data in parts:
        if len(data) == 0:
            continue
        # minimalizacja krawędzi przejścia za pomocą okna Hanninga
        window = np.hanning(len(data))
        data = data * window
        fft_v = abs(fft.fft(data)) / sample_rate
        fft_r = copy(fft_v)
        for i in range(2, HPSLoop):
            tab = copy(fft_v[::i])
            fft_r = fft_r[:len(tab)]
            fft_r *= tab
        result_parts.append(fft_r)
    result = [0] * len(result_parts[int(len(result_parts) / 2)])
    for res in result_parts:
        if len(res) != len(result):
            continue
        result += res

    if sum(result[male_min_max[0]: male_min_max[1]]) > sum(result[female_min_max[0]: female_min_max[1]]):
        return 0
    return 1


def true_gender(file):
    # wyodrębnienie oryginalnej płci z nazwy pliku
    if file.replace("/", "_").replace(".", "_").split("_")[1] == "M":
        return 0
    else:
        return 1


if __name__ == "__main__":
    M = [[0, 0], [0, 0]]
    files = glob.glob("005_M.wav")
    for f in files:
        sr, array = wavfile.read(f)
        correct = true_gender(f)    # dla mężczyzny: 0 dla kobiety: 1
        found = harmonic_product_spectrum(sr, array)
        M[correct][found] += 1
    print(M)
    correctness = (M[0][0] + M[1][1]) / (sum(M[0]) + sum(M[1]))
    print(correctness)
