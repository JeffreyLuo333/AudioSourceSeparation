# Audio Source Separation

Audio Source Separation involves distinguishing a composite sound (e.g., a concerto) into its individual components (e.g., just the lead vocals). In my current project, I aim to delve into audio signal representation and investigate the potential of using AI models for sound separation. 

Project details can be found here: [Audio Source Separation on GitHub](https://github.com/JeffreyLuo333/Audio-Source-Separation).

For dataset creation, I merged my piano recordings with other violin sound sources. My initial approach utilizes the FNN model, focusing on training with the sound wave format. 

- **Exploration with FNN**: [Piano Sound Wave with FNN](https://github.com/JeffreyLuo333/Audio-Source-Separation/blob/main/notebooks/PianoSoundWaveFNN.ipynb).

Additionally, I'm keen on evaluating the FNN model's performance with spectrograms. 

- **Preliminary Work on Conversion**: [Wave to Spectrogram Conversion](https://github.com/JeffreyLuo333/Audio-Source-Separation/blob/main/notebooks/WaveSpectrogramConversion.ipynb).

Moving forward, I plan to use spectrograms as training data to gauge the results further.
