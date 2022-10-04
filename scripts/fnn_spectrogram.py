import os
import math
import librosa
import numpy as np
import keras
from keras.layers import Dense, Dropout, PReLU
from keras.optimizers import Adam
import soundfile as sf
import pickle

DATASET_PATH = "./data/audio/train/piano/10"
MODEL_PATH = os.path.join(DATASET_PATH, "model")

FFT_FRAME_LENGTH = 2048
WAVE_SAMPLE_RATE = 44100
HOP_LENGTH = 512
TRACK_DURATION = 60 # measured in seconds
SAMPLES_PER_TRACK = WAVE_SAMPLE_RATE * TRACK_DURATION
NUM_SEGMENTS = 12

def get_TSpectrogram_from_Wave(wave_file_path, data_dir, num_segments=NUM_SEGMENTS, sr = WAVE_SAMPLE_RATE, n_fft=FFT_FRAME_LENGTH, hop_length=HOP_LENGTH):
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    filename = os.path.basename(wave_file_path)
    f_token = filename.split('.')[0]
    data_file_name = f_token + '.pkl'

    # create segment directory
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    dataset_path = os.path.join(data_dir, data_file_name)
    if os.path.exists(dataset_path):
        # retrieve the data and return
        file = open(dataset_path, 'rb')
        print("----- getting dataset from {}".format(dataset_path))
        TSpec = pickle.load(file)
        file.close()
        return TSpec

    dataset = []
    signal, sr = librosa.load(wave_file_path, sr=sr)
            
    # process all segments of audio file
    for d in range(num_segments):
        # calculate start and finish sample for current segment
        start = samples_per_segment * d
        finish = start + samples_per_segment

        # extract stft
        S_signal = librosa.stft(signal[start:finish], n_fft=n_fft, hop_length=hop_length)
        Y_signal = np.abs(S_signal) ** 2
        Y_log_signal = librosa.power_to_db(Y_signal)
        TSpec_signal = Y_log_signal.T # y: freuency, x: time vector

        f_vectors = TSpec_signal.shape[1]
        dataset.append(TSpec_signal.tolist())

    TSpec = np.array(dataset).reshape(-1,f_vectors)

    print(TSpec.shape)

    # save to pickle file
    file = open(dataset_path, 'wb')
    print("----- generating dataset {}".format(dataset_path))
    pickle.dump(TSpec, file)
    file.close()
    return TSpec


def create_TSpectrogram(dataset_path, data_output_dir, num_segments=NUM_SEGMENTS, n_fft=FFT_FRAME_LENGTH, hop_length=HOP_LENGTH):    
    if not os.path.exists(data_output_dir):
        os.makedirs(data_output_dir)

    for dirpath, dirnames, filenames in os.walk(dataset_path):
        if dirpath ==  data_output_dir:
            print('----- skipping {} {}'.format(dirpath, data_output_dir))
            continue


        print('----- processing {}'.format(dirpath))
        # process all audio files in genre sub-dir
        for f in filenames:
            # load all audio file
            if "mix" in f:
                m_wave_file_path = os.path.join(dataset_path, f)
                m_TSpec = get_TSpectrogram_from_Wave(m_wave_file_path, data_output_dir, num_segments=num_segments, sr = WAVE_SAMPLE_RATE, n_fft=n_fft, hop_length=hop_length)

                t_token = f.split('_')[2]
                t_file = t_token + '.wav'
                t_wave_file_path = os.path.join(dirpath,  t_file)
                t_TSpec = get_TSpectrogram_from_Wave(t_wave_file_path, data_output_dir, num_segments=num_segments, sr = WAVE_SAMPLE_RATE, n_fft=n_fft, hop_length=hop_length)
    return


def create_traing_dataset(dataset_path, dataset_files):   
    F_input_file_path = os.path.join(dataset_path, 'input.pkl')
    F_target_file_path = os.path.join(dataset_path,'target.pkl')

    input_TSpec = np.array([])
    target_TSpec = np.array([])
    input_dataset = []
    target_dataset = []
        
    if os.path.exists(F_input_file_path) and os.path.exists(F_input_file_path) :
        print("----- getting input dataset from {}".format(F_input_file_path))
        file = open(F_input_file_path, 'rb')
        input_TSpec = pickle.load(file)
        file.close()

        print("----- getting target dataset from {}".format(F_target_file_path))
        file = open(F_target_file_path, 'rb')
        target_TSpec = pickle.load(file)
        file.close()
        return input_TSpec, target_TSpec
    

    for f in dataset_files:
        input_file_path = os.path.join(dataset_path, f)
        print("----- getting input dataset from {}".format(input_file_path))
        file = open(input_file_path, 'rb')
        TSpec = pickle.load(file)
        file.close()
        input_dataset.append(TSpec.tolist())

        t_token = f.split('_')[2]
        tf = t_token + '.pkl'
        target_file_path = os.path.join(dataset_path, tf)
        print("----- getting target dataset from {}".format(target_file_path))
        file = open(target_file_path, 'rb')
        TSpec = pickle.load(file)
        file.close()
        f_vectors = TSpec.shape[1]
        target_dataset.append(TSpec.tolist())

    input_TSpec = np.array(input_dataset).reshape(-1, f_vectors)
    target_TSpec = np.array(target_dataset).reshape(-1, f_vectors)

    #save to pickle file
    file = open(F_input_file_path, 'wb')
    print("----- generating input dataset {}".format(F_input_file_path))
    pickle.dump(input_TSpec, file)
    file.close()

    file = open(F_target_file_path, 'wb')
    print("----- generating target dataset {}".format(F_target_file_path))
    pickle.dump(target_TSpec, file)
    file.close()

    return input_TSpec, target_TSpec


if __name__ == "__main__":

    if not os.path.exists(MODEL_PATH):
        OUTPUT_PATH = os.path.join(DATASET_PATH, 'pkl')
        create_TSpectrogram(DATASET_PATH, OUTPUT_PATH, num_segments=NUM_SEGMENTS, n_fft=FFT_FRAME_LENGTH, hop_length=HOP_LENGTH)  
        input_list = ['mix_10_piano1_10_violin1.pkl', 'mix_10_piano1_10_violin2.pkl', 'mix_10_piano1_10_violin3.pkl', 'mix_10_piano2_10_violin1.pkl', 'mix_10_piano2_10_violin2.pkl', 'mix_10_piano2_10_violin3.pkl']
        input, target = create_traing_dataset(OUTPUT_PATH, input_list)

        # create model
        input_shape = input[0].shape
        input_size = input[0].shape[0]
        output_size = target[0].shape[0]

        samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
        num_vectors_per_segment = math.ceil(samples_per_segment / HOP_LENGTH)

        model = keras.models.Sequential()
        model.add(Dense(input_size, input_shape=input_shape))
        model.add(PReLU())
        model.add(Dense(512))
        model.add(PReLU())
        model.add(Dense(output_size))
        model.compile(Adam(), 'mse')
        model.summary()

        history = model.fit(input, target, epochs = 50)
        model.save(MODEL_PATH)
    else:
        model = keras.models.load_model(MODEL_PATH)
    
    test_TSpec = get_TSpectrogram_from_Wave("./data/audio/train/piano/10/piano2.wav", 
                                            "./data/audio/train/piano/10/pkl",
                                            num_segments=12)
    Spec_signal = librosa.db_to_amplitude(test_TSpec.T)
    A_signal = librosa.istft(Spec_signal, hop_length=HOP_LENGTH)
    A_signal_max = max(A_signal)
    A_signal_adjust = A_signal/A_signal_max 
    sf.write("S_piano2.wav", A_signal_adjust, WAVE_SAMPLE_RATE)
    
    test_TSpec = get_TSpectrogram_from_Wave("./data/audio/train/piano/10/mix_10_piano2_10_violin1.wav", 
                                            "./data/audio/train/piano/10/pkl",
                                            num_segments=12)
    Spec_signal = librosa.db_to_amplitude(test_TSpec.T)
    A_signal = librosa.istft(Spec_signal, hop_length=HOP_LENGTH)
    A_signal_max = max(A_signal)
    A_signal_adjust = A_signal/A_signal_max 
    sf.write("S_mix_10_piano2_10_violin1.wav", A_signal_adjust, WAVE_SAMPLE_RATE)

    predict_data = model.predict(test_TSpec.tolist())
    Spec_signal = librosa.db_to_amplitude(predict_data.T)
    A_signal = librosa.istft(Spec_signal, hop_length=HOP_LENGTH)
    A_signal_max = max(A_signal)
    A_signal_adjust = A_signal/A_signal_max 
    sf.write("S_predict.wav", A_signal_adjust, WAVE_SAMPLE_RATE)
