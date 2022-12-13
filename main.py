import glob
from scipy.io import wavfile
from pylab import *
from scipy import *

male_min_max = [55, 158]
female_min_max = [170, 295]
HPSLoop = 6
samples_number = 3  # Branie pod uwagę tylko 3 1-sekundowych próbek


def read_files_from_dir(location):
    files = glob.glob(location)
    return files


def true_gender(file):
    # wyodrębnienie oryginalnej płci z nazwy pliku
    if file.replace("/", "_").replace(".", "_").split("_")[1] == "M":
        return 0
    else:
        return 1


def read_wav(file):
    sample_rate, data = wavfile.read(file)
    return sample_rate, data


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
    sample_array = [[0 for column in range(part_length)] for row in range(int(samples_counter))]
    for i in range(int(samples_counter)):
        sample_array[i] = data_voice[i * part_length: (i + 1) * part_length]
    return sample_array


def apply_hanning(data):
    window = np.hanning(len(data))
    data = data * window
    return data


def harmonic_product_spectrum(sample_rate, sample_array):
    partial_result = []
    for data in sample_array:
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
        partial_result.append(fft_r)
    result = [0] * len(partial_result[int(len(partial_result) / 2)])
    for res in partial_result:
        #    if len(res) != len(result):
        #        continue
        result += res
    return result


def check_gender(result):
    if sum(result[male_min_max[0]: male_min_max[1]]) > sum(result[female_min_max[0]: female_min_max[1]]):
        return 0
    return 1


def fill_and_print_coverage_matrix(files):
    m = [[0, 0], [0, 0]]
    for wav in files:
        sample_rate, data = read_wav(wav)
        correct_gender = true_gender(wav)  # dla mężczyzny: 0 dla kobiety: 1
        found_gender = harmonic_product_spectrum(sample_rate, slice_into_samples(samples_number, data, sample_rate))
        m[correct_gender][check_gender(found_gender)] += 1
    print(m)
    correctness_percent = (m[0][0] + m[1][1]) / (sum(m[0]) + sum(m[1]))
    print(correctness_percent)


def find_and_print_gender(wav):
    sample_rate, data = read_wav(wav)
    found_gender = harmonic_product_spectrum(sample_rate, data, slice_into_samples(samples_number, data, sample_rate))
    if check_gender(found_gender) == 0:
        print('M')
    elif check_gender(found_gender) == 1:
        print('K')


if __name__ == "__main__":
    '''
    if len(sys.argv) != 2:
        print("Incorrect number of arguments. Please enter path to one .wav file.")
        exit(1)
    path = Path(sys.argv[1])
    if path.suffix != '.wav':
        print("Incorrect file extension. Please enter path to .wav file.")
        exit(1)
    find_and_print_gender(path)
    '''
    path = "trainall/*"
    files_array = read_files_from_dir(path)
    fill_and_print_coverage_matrix(files_array)
