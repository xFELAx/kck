import glob

import numpy as np
from scipy.io import wavfile
from pylab import *
from scipy import *

male_min_max = [55, 158]
female_min_max = [170, 295]
HPSLoop = 6


def slice_into_samples(samples_counter, data_voice, sample_rate):
    if samples_counter > len(data_voice) / sample_rate:
        samples_counter = len(data_voice) / sample_rate
    data_voice = data_voice[
                 max(0, int(len(data_voice) / 2) - int(samples_counter / 2 * sample_rate)): min(len(data_voice) - 1,
                                                                                               int(len(
                                                                                                   data_voice) / 2) + int(
                                                                                                   samples_counter / 2 * sample_rate))]
    # Branie pod uwagę tylko 3 1-sekundowych próbek z analizowanej tablicy
    part_length = int(sample_rate)

    # Podział tablicy na 3 1-sekundowe próbki
    # parts1 = [data_voice[i * part_len: (i + 1) * part_len] for i in range(int(samples_counter))]
    samples_array = [[0 for column in range(part_length)] for row in range(int(samples_counter))]
    for i in range(int(samples_counter)):
        samples_array[i] = data_voice[i * part_length: (i + 1) * part_length]
    return samples_array


def check_gender(result):
    if sum(result[male_min_max[0]: male_min_max[1]]) > sum(result[female_min_max[0]: female_min_max[1]]):
        return 0
    return 1


def apply_hanning(data):
    window = np.hanning(len(data))
    data = data * window
    return data


def harmonic_product_spectrum(sample_rate, data_voice, samples_array):
    result_parts = []
    for data in samples_array:
        # if len(data) == 0:
        #    continue
        # minimalizacja krawędzi przejścia za pomocą okna Hanninga
        windowed_data = apply_hanning(data)
        fft_v = abs(fft.fft(windowed_data)) / sample_rate
        fft_r = copy(fft_v)
        # kompresja spektrum
        for i in range(2, HPSLoop):
            tab = copy(fft_v[::i])
            fft_r = fft_r[:len(tab)]
            fft_r *= tab
        result_parts.append(fft_r)
    result = [0] * len(result_parts[int(len(result_parts) / 2)])
    for res in result_parts:
        #    if len(res) != len(result):
        #        continue
        result += res
    return result


def true_gender(file):
    # wyodrębnienie oryginalnej płci z nazwy pliku
    if file.replace("/", "_").replace(".", "_").split("_")[1] == "M":
        return 0
    else:
        return 1


if __name__ == "__main__":
    M = [[0, 0], [0, 0]]
    files = glob.glob("trainall/*")
    samples_number = 3  # Branie pod uwagę tylko 3 1-sekundowych próbek
    for f in files:
        sr, array = wavfile.read(f)
        correct = true_gender(f)  # dla mężczyzny: 0 dla kobiety: 1
        found = harmonic_product_spectrum(sr, array, slice_into_samples(samples_number, array, sr))
        M[correct][check_gender(found)] += 1
    print(M)
    correctness = (M[0][0] + M[1][1]) / (sum(M[0]) + sum(M[1]))
    print(correctness)
