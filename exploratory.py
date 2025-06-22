import os
import numpy as np
import random
from librosa import feature, power_to_db, load, display
import matplotlib.pyplot as plt


def plot_autocorrelation(signal, max_lag=100, title='Autocorelatie'):
    signal = signal - np.mean(signal)
    result = np.correlate(signal, signal, mode='full')
    mid = len(result) // 2
    ac = result[mid:mid + max_lag]
    ac /= ac[0]  # normalize
    plt.plot(ac)
    plt.title(title)
    plt.xlabel("Intarziere")
    plt.ylabel("Autocorelatie")
    plt.grid()
    plt.show()


random.seed(42)

path = "./DCASE2024"

machine_folders = os.listdir(path)

for machine_name in machine_folders:
    datasetType, machineType = machine_name.split('_')
    recordings_folder = os.path.join(path, machine_name, machineType)
    machines_test_folder = os.path.join(recordings_folder, 'test')
    machines_test_recordings = [file for file in os.listdir(machines_test_folder) if file.endswith(".wav")]
    random.shuffle(machines_test_recordings)

    audio_file, sr = load(os.path.join(machines_test_folder, machines_test_recordings[0]), sr=16000)
    if len(audio_file) < sr*10:
        audio_file = np.pad(audio_file, (0, sr*10 - len(audio_file)), constant_values=(0, 0))
    else:
        audio_file = audio_file[:sr*10]

    if machines_test_recordings[0].find("anomaly") != -1:
        print("Signal is anomalous!")
    else:
        print("Signal is normal!")
    timesteps = np.arange(0, 10, 1/16000)
    print(f"Min value audio file:{np.min(audio_file)}")
    print(f"Max value audio file:{np.max(audio_file)}")
    print(f"Mean value audio file:{np.mean(audio_file)}")
    print(f"Median value audio file:{np.median(audio_file)}")

    mel_spec = feature.melspectrogram(y=audio_file, sr=sr, n_fft=1024, hop_length=352, n_mels=64)
    db_mel_spec = power_to_db(mel_spec)

    print(f"Shape of spectrogram:{db_mel_spec.shape}")
    print(f"Min value spectrogram:{np.min(db_mel_spec)}")
    print(f"Max value spectrogram:{np.max(db_mel_spec)}")
    print(f"Mean value spectrogram:{np.mean(db_mel_spec)}")
    print(f"Median value spectrogram:{np.median(db_mel_spec)}")
    plt.plot(timesteps, audio_file)
    plt.title("Audio signal waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()
    plt.title("Histograma semnalului audio")
    plt.xlabel("Amplitudine")
    plt.ylabel("Numar de esantioane")
    plt.hist(audio_file, bins=160)
    plt.show()
    fig, ax = plt.subplots()
    img = display.specshow(db_mel_spec, x_axis='time', y_axis="mel", sr=16000, fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()
    for index in [0, 7, 15]:
        plt.title(f"Mel Band {index} Histogram")
        plt.xlabel("Power")
        plt.ylabel("Num of Samples")
        plt.hist(db_mel_spec[index, :], bins=15)
        plt.show()

    # avg_band = np.mean(db_mel_spec[24:44], axis=0)  # mid-range mel bands

    plot_autocorrelation(db_mel_spec[20], max_lag=100, title='Functia de autocorelatie a unei benzi de frecventa mijlocie')

    # Correlation Matrix of Mel Bands
    corr_matrix = np.corrcoef(db_mel_spec)
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix)
    plt.colorbar()
    plt.title("Matricea de corelatie a benzilor mel")
    plt.show()

    normed_mel_spec = (db_mel_spec - np.min(db_mel_spec))/(np.max(db_mel_spec) - np.min(db_mel_spec))
    print(f"Min value spectrogram:{np.min(normed_mel_spec)}")
    print(f"Max value spectrogram:{np.max(normed_mel_spec)}")
    print(f"Mean value spectrogram:{np.mean(normed_mel_spec)}")
    print(f"Median value spectrogram:{np.median(normed_mel_spec)}")
    fig, ax = plt.subplots()
    img = display.specshow(normed_mel_spec, x_axis='time', y_axis="mel", sr=16000, fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()
    for index in [0, 7, 15]:
        plt.title(f"Mel Band {index+1} Histogram")
        plt.xlabel("Power")
        plt.ylabel("Num of Samples")
        plt.hist(normed_mel_spec[index, :], bins=15)
        plt.show()

    # Correlation Matrix of Mel Bands
    corr_matrix = np.corrcoef(normed_mel_spec)
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix)
    plt.colorbar()
    plt.title("Mel Band Correlation Matrix")
    plt.show()


