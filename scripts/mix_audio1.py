import os
import librosa
import numpy as np
import soundfile as sf
 
SAMPLE_RATE = 44100
DURATION = 60 # 60 seconds
HOP_RATE = 512

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
            norm_signal = normalise(signal, max=1, min=-1)
            sf.write(output_file, norm_signal, sample_rate)

def create_mix_files(audio_files_dir, inst1, inst2, sample_rate=SAMPLE_RATE, duration=DURATION, segment_duration = 5) :
    instruments = []
    instruments.append(inst1)
    instruments.append(inst2)

    samples_per_segments = sample_rate * segment_duration
    segments = int(duration / segment_duration)
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
            mix_signal = (x_signal + y_signal)
            
            for d in range(segments) :
                x_file_name = x_name + "_" + str(d) + ".wav"
                x_file_path = os.path.join(audio_files_dir, x_file_name)
                mix_file_name = "mix_" + x_name + "_" + y_name + "_" + str(d) + ".wav"
                mix_file_path = os.path.join(audio_files_dir, mix_file_name)
                print(f"- processing input file {mix_file_path}")
                start = start = samples_per_segments * d
                finish = start + samples_per_segments
                sf.write(mix_file_path, mix_signal[start:finish], sample_rate) 
                sf.write(x_file_path, x_signal[start:finish], sample_rate) 

    for x in instrument_file_list[0]:
        x_path = os.path.join(audio_files_dir, x)
        os.remove(x_path)
    
    for y in instrument_file_list[1]:
        y_path = os.path.join(audio_files_dir, y)
        os.remove(y_path)


if __name__ == "__main__":
    DATASET_DIR = "./data/autoencoder_run"
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
                    