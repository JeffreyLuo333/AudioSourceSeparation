import os
import librosa
import numpy as np
import soundfile as sf
 
AUDIO_DIR = "./data/audio/"
SAMPLE_RATE = 44100
#SAMPLE_RATE = 22050

def create_instrument_files(instrument, signal_level_list, sample_rate = SAMPLE_RATE):
    list = []
    instrument_path = os.path.join(AUDIO_DIR, "instrument", instrument)
    convert_path = os.path.join(AUDIO_DIR, "train", instrument)

    if not os.path.exists(convert_path):
        os.makedirs(convert_path)

    for (root, dirs, files) in os.walk(instrument_path):
        for f in files:
            if '.wav' in f:
                # read the audio signal
                input_file = os.path.join(instrument_path, f)
                signal, sr = librosa.load(input_file, sr = sample_rate)
                print("- processing input file {}.".format(input_file))
                signal_max = max(signal)
                signal_10 = signal / signal_max
                for i in signal_level_list:
                    c_convert_path_signal_level = os.path.join(convert_path, str(i))
                    if not os.path.exists(c_convert_path_signal_level):
                        os.makedirs(c_convert_path_signal_level)
                    output_file = os.path.join(c_convert_path_signal_level, f)

                    output_signal = signal_10 * 0.1 * i
                    sf.write(output_file, output_signal, sr)
                    print ("    - creating output file {}.".format(output_file))


def create_mix_files(t_instrument, c_instruments, sample_rate = SAMPLE_RATE, t_signal_level = 10, c_signal_level = 10): 
    t_instrument_path = os.path.join(AUDIO_DIR, 'train', t_instrument)
    
    # dirs=directory
    for c_instrument in c_instruments:
        c_instrument_path = os.path.join(AUDIO_DIR, 'train', c_instrument)

        f_target_signal = []
        f_mix_signal = []

        t_input_path = os.path.join(t_instrument_path, str(t_signal_level))
        
        for (troot, tdirs, tfile) in os.walk(t_input_path):
            for tf in tfile:
                if "mix" in tf:
                    continue
            
                t_token = tf.split('.')[0]
                t_file_path = os.path.join(t_instrument_path, str(t_signal_level) , tf)
                t_signal, sr = librosa.load(t_file_path, sr = sample_rate)

                c_file_dir_path = os.path.join(c_instrument_path, str(c_signal_level))
                for (croot, cdirs, cfile) in os.walk(c_file_dir_path):
                    for cf in cfile:
                        if "mix" in cf:
                            continue

                        c_token = cf.split('.')[0]
                        c_file_path = os.path.join(c_file_dir_path, cf)
                        c_signal, sr = librosa.load(c_file_path, sr = sample_rate)
                        m_signal = t_signal + c_signal
                        mix_file = 'mix_' + str(t_signal_level) + '_' + t_token + '_' + str(c_signal_level) + '_' + c_token + '.wav'
                        mix_file_path = os.path.join(t_input_path, mix_file)
                        print('- creating {}.'. format(mix_file_path))
                        sf.write(mix_file_path, m_signal, sample_rate)    

if __name__ == "__main__":
    create_instrument_files('piano', [10])
    create_instrument_files('violin', [10])
    create_mix_files('piano', ['violin'])
                    