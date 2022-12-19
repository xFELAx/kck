from scipy.io import wavfile
from pylab import *
from scipy import *
import numpy as np
from os import listdir
from os.path import isfile, join

MAX_AUDIO_DURATION = 5

MALE_MAX_FREQ = 158
MALE_MIN_FREQ = 55

FEMALE_MIN_FREQ = 170
FEMALE_MAX_FREQ = 295


def slice_audio(data, sample_rate):
    audio_duration = min(len(data) / sample_rate, MAX_AUDIO_DURATION)
    samples_array = []
    for i in range(int(audio_duration)):
        samples_array.append(data[i * sample_rate: (i + 1) * sample_rate])
    return samples_array


def classify(spectrum):
    male_freq_range = MALE_MAX_FREQ - MALE_MIN_FREQ
    male_compliance = sum(
        spectrum[MALE_MIN_FREQ:MALE_MAX_FREQ]) / male_freq_range

    female_freq_range = FEMALE_MAX_FREQ - FEMALE_MIN_FREQ
    female_compliance = sum(
        spectrum[FEMALE_MIN_FREQ:FEMALE_MAX_FREQ]) / female_freq_range

    return female_compliance > male_compliance


def apply_hanning(data):
    window = np.hanning(len(data))
    data = data * window
    return data


def harmonic_product_spectrum(sample_rate, samples_array):
    result_parts = []
    for data in samples_array:
        windowed_data = apply_hanning(data)
        fft_v = abs(fft.fft(windowed_data)) / sample_rate
        fft_r = fft_v.copy()
        for i in range(2, 6):
            tab = fft_v[::i].copy()
            fft_r = fft_r[:len(tab)]
            fft_r *= tab
        result_parts.append(fft_r)
    result = [0] * len(result_parts[int(len(result_parts) / 2)])
    for res in result_parts:
        result += res
    return result


def get_actual_gender(filename):
    return "K" in filename


def read_files_from_dir(path):
    return [path + f for f in listdir(path) if isfile(join(path, f))]


def predict_gender(files):
    predicted = []
    actual = []
    for wav in files:
        actual.append(get_actual_gender(wav))
        sample_rate, data = wavfile.read(wav)
        audio_slices = slice_audio(data, sample_rate)
        hps_result = harmonic_product_spectrum(sample_rate, audio_slices)
        predicted.append(classify(hps_result))
    return predicted, actual


if __name__ == "__main__":
    path = "./trainall/"
    files = read_files_from_dir(path)
    predicted, actual = predict_gender(files)
    correct_predictions = 0
    for i in range(0, len(files)):
        if predicted[i] == actual[i]:
            correct_predictions += 1
    print(correct_predictions / len(files))
