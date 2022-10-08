import os
import librosa
import numpy as np
import soundfile as sf
 
SAMPLE_RATE = 44100
DURATION = 60 # 60 seconds

def normalise(array, max=1, min=0):
    norm_array = (array - array.min()) / (array.max() - array.min())
    norm_array = norm_array * (max - min) + min
    return norm_array

def create_instrument_files(audio_files_dir, signal_files_dir, sample_rate=SAMPLE_RATE, duration=DURATION) :
    list = []
    for root, _, files in os.walk(audio_files_dir):
        for f in files:
            # read the audio signal
            input_file = os.path.join(audio_files_dir, f)
            output_file = os.path.join(signal_files_dir, f)
            print(f"- processing input file {input_file}.")
            signal = librosa.load(input_file, sr=sample_rate, duration=duration, mono=True)[0]
            norm_signal = normalise(signal)
            sf.write(output_file, norm_signal, sample_rate)

def create_mix_files(audio_files_dir, inst1, inst2, sample_rate=SAMPLE_RATE, duration=DURATION) :
    instruments = []
    instruments.append(inst1)
    instruments.append(inst2)

    instrument_file_list = []
    for instrument in instruments:
        file_list = []
        for root, _, files in os.walk(audio_files_dir):
            for file in files:
                if "mix" in file:
                    continue
                if instrument in file:
                    file_list.append(file)
        instrument_file_list.append(file_list)

    for x in instrument_file_list[0]:
        x_path = os.path.join(audio_files_dir, x)
        x_signal = librosa.load(x_path, sr=sample_rate, duration=duration, mono=True)[0]
        x_name= x.split('.')[0]
        for y in instrument_file_list[1]:
            y_path = os.path.join(audio_files_dir, y)
            y_signal = librosa.load(y_path, sr=sample_rate, duration=duration, mono=True)[0]
            y_name = y.split('.')[0]
            mix_file_name = "mix_" + x_name + "_" + y_name + ".wav"
            norm_signal = normalise(x_signal + y_signal)
            mix_file_path = os.path.join(audio_files_dir, mix_file_name)
            print(f"- processing input file {mix_file_path}")
            sf.write(mix_file_path, norm_signal, sample_rate)     

if __name__ == "__main__":
    DATASET_DIR = "./data/fnn_run"
    SIGNAL_SAVE_DIR = os.path.join(DATASET_DIR, "audio")
    INSTRUMENT_DIR = "./data/audio/instrument/"

    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    if not os.path.exists(SIGNAL_SAVE_DIR):
        os.makedirs(SIGNAL_SAVE_DIR)

    instruments = ["piano", "violin"]
    for instrument in instruments:
        instrument_path = os.path.join(INSTRUMENT_DIR, instrument)
        create_instrument_files(instrument_path, SIGNAL_SAVE_DIR, sample_rate=SAMPLE_RATE, duration=DURATION)

    create_mix_files(SIGNAL_SAVE_DIR, "piano", "violin", sample_rate=SAMPLE_RATE)
                    