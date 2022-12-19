from scipy.io import wavfile
from scipy import *
import numpy as np
from os import listdir
from os.path import isfile, join
import sys

FEMALE_CUTOFF_FREQUENCY = 175
HUMAN_MIN_FREQENCY = 40
HUMAN_MAX_FREQUENCY = 600
HPS_ITERATIONS = 5
MAX_AUDIO_DURATION = 5


def predict_gender(file):
    sample_rate, data = wavfile.read(file)
    audio_slices = slice_audio(data, sample_rate)
    hps_result = harmonic_product_spectrum(sample_rate, audio_slices)
    return classify(hps_result)


def slice_audio(data, sample_rate):
    audio_duration = min(len(data) / sample_rate, MAX_AUDIO_DURATION)
    samples_array = []
    for i in range(int(audio_duration)):
        samples_array.append(data[i * sample_rate: (i + 1) * sample_rate])
    return samples_array


def harmonic_product_spectrum(sample_rate, audio_slices):
    hp_array = []
    for slice in audio_slices:
        windowed_data = apply_hanning(slice)
        spectrum = abs(fft.fft(windowed_data)) / sample_rate
        harmonic_product = spectrum.copy()
        for i in range(2, HPS_ITERATIONS):
            subset = spectrum[::i]
            harmonic_product = harmonic_product[:len(subset)]
            harmonic_product *= subset
        hp_array.append(harmonic_product)
    return sum(hp_array)


def apply_hanning(data):
    window = np.hanning(len(data))
    data = data * window
    return data


def classify(spectrum):
    human_spectrum = spectrum[HUMAN_MIN_FREQENCY:HUMAN_MAX_FREQUENCY]
    dominant_frequency = spectrum.tolist().index(max(human_spectrum))
    return "K" if dominant_frequency > FEMALE_CUTOFF_FREQUENCY else "M"


if __name__ == "__main__":
    try:
        input_file = sys.argv[1]
        predicted = predict_gender(input_file)
        print(predicted)
    except:
        print("M")
