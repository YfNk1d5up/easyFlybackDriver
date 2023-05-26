import wave

import essentia.standard as ess
import pandas as pd
import os
import numpy
import matplotlib.pyplot as plt

# we start by instantiating the audio loader:
loader = ess.MonoLoader(filename='separated/htdemucs/test-Polyphia-G.O.A.T/other.wav', )


multipitchExtractor = ess.MultiPitchMelodia()
keyExtractor = ess.KeyExtractor()
rhythmExtractor = ess.RhythmExtractor()
sineModelAnal = ess.SineModelAnal()
filter = ess.BandPass(
    bandwidth=700,
    cutoffFrequency=880,
    sampleRate=44100
)
sineModelSynth = ess.SineModelSynth()
ifft = ess.IFFT()
overl = ess.OverlapAdd()
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

import synthetize_notes
FFT_WINDOWS_SECONDS = 128 / 44100

synthetize_notes.save_wav_multi('multipitchmelodia', TOP_NOTES=4, sampling_rate=16000, FFT_WINDOW_SECONDS=FFT_WINDOWS_SECONDS, amplitude=1000)