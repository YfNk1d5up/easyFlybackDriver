import wave

import essentia.standard as ess
import pandas as pd
import numpy as np
import os
import tqdm
import numpy
import matplotlib.pyplot as plt


# input and output files
inputFilename = 'other.wav'
outputFilename = 'synthesis_sinemodel.wav'
# algorithm parameters
params = {
    "frameSize": 2048,
    "hopSize": 512,
    "startFromZero": False,
    "sampleRate": 44100,
    "maxnSines": 2,
    "magnitudeThreshold": -74,
    "minSineDur": 0.02,
    "freqDevOffset": 10,
    "freqDevSlope": 0.001,
}

# we start by instantiating the audio loader:
loader = ess.MonoLoader(filename='separated/htdemucs/test-Polyphia-G.O.A.T/other.wav', )


multipitchExtractor = ess.MultiPitchMelodia(
    frameSize=params["frameSize"],
    hopSize=params["hopSize"],
    sampleRate=params["sampleRate"],
    minFrequency=180,
    maxFrequency=880,
    magnitudeThreshold= -47
)
keyExtractor = ess.KeyExtractor()
rhythmExtractor = ess.RhythmExtractor()
sineModelAnal = ess.SineModelAnal()
filter = ess.BandPass(
    bandwidth=700,
    cutoffFrequency=880,
    sampleRate=44100
)
smsynstd = ess.SineModelSynth(
    sampleRate=params["sampleRate"],
    fftSize=params["frameSize"],
    hopSize=params["hopSize"],
)
ifftstd = ess.IFFT(size=params["frameSize"])
overlstd = ess.OverlapAdd(frameSize=params["frameSize"], hopSize=params["hopSize"])
awritestd = ess.MonoWriter(
    filename=str(outputFilename), sampleRate=params["sampleRate"]
)
awrite = [[],[],[],[]]
for i in range(4):
    awrite[i] = ess.MonoWriter(
        filename=f'data/output/multipitch/4voices_synthetize{i}.wav')
# print(keyExtractor(audio))
# print(rhythmExtractor(audio)[0])
# and then we actually perform the loading:
audio = loader()
audio = ess.EqualLoudness()(audio)
audio = filter(audio)
print(len(audio) / 44100.0)
pitch_curve = multipitchExtractor(audio)
n_frames = len(pitch_curve)
print("number of frames: %d" % n_frames)

audioout = np.array(0)
# loop over all frames
for freq in tqdm.tqdm(pitch_curve):
    magn = []
    phase = []
    for f in freq:
        magn.append(1)
        phase.append(0)
    outfft = smsynstd(np.array(magn),
                   np.array(freq),
                   np.array(phase)
                   )

    # STFT synthesis
    out = overlstd(ifftstd(outfft))
    audioout = np.append(audioout, out)


# write audio output
awritestd(audioout.astype(np.float32))







"""

# Plot the estimated pitch contour and confidence over time.f, axarr = plt.subplots(2, sharex=True)
note_lst = []
for i in range(n_frames):
    timeframe_notes = [float(i * 128 / 44100)]
    for j in range(4):
        freq = 0
        try:
            freq = pitch_curve[i][j]
        except:
            pass
        timeframe_notes.append(freq)
        timeframe_notes.append(1)
    note_lst.append(timeframe_notes)
    # fig = plot_fft(fft.real, xf, fs, s, RESOLUTION)    # fig.write_image(f"data/frames/frame{frame_number}.png", scale=2)columns = ["time"]
columns = ["time"]
for n in range(4):
    columns.append(f"note{n}")
    columns.append(f"anote{n}")
note_table = pd.DataFrame(note_lst, columns=columns)
note_table.to_csv(os.path.join("data/output/",'multipitchmelodia.csv'),index=False)


"""
"""
import synthetize_notes
FFT_WINDOWS_SECONDS = 128 / 44100

synthetize_notes.save_wav_multi('multipitchmelodia', TOP_NOTES=4, sampling_rate=16000, FFT_WINDOW_SECONDS=FFT_WINDOWS_SECONDS, amplitude=1000)
"""