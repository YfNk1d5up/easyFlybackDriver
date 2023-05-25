import essentia
import essentia.standard as ess
import numpy
import matplotlib.pyplot as plt
import libf0.utils as sonify

# we start by instantiating the audio loader:
loader = ess.MonoLoader(filename='separated/htdemucs/test-Polyphia-G.O.A.T/other.wav', )

# and then we actually perform the loading:
audio = loader()
audio = ess.EqualLoudness()(audio)
print(len(audio)/44100.0)
multipitchExtractor = ess.MultiPitchMelodia()
keyExtractor = ess.KeyExtractor()
rhythmExtractor = ess.RhythmExtractor()
sineModelSynth = ess.SineModelSynth()
#print(keyExtractor(audio))
#print(rhythmExtractor(audio)[0])
pitch_curve = multipitchExtractor(audio)
n_frames = len(pitch_curve)
print("number of frames: %d" % n_frames)
# Pitch is estimated on frames. Compute frame time positions.
pitch_times = numpy.linspace(0.0,len(audio)/44100.0,len(pitch_curve) )
multipitch_curve = [[],[],[],[]]
multipitch_curve_amp = [[],[],[],[]]
multipitch_curve_phase = [[],[],[],[]]
for f in pitch_curve:
    for i in range(4):
        try:
            multipitch_curve[i].append(f[i])
            multipitch_curve_amp[i].append(1)
            multipitch_curve_phase[i].append(0)
        except:
            multipitch_curve[i].append(0)
# Plot the estimated pitch contour and confidence over time.
f, axarr = plt.subplots(2, sharex=True)
fft_sig = []
for i in range(4):
    #xtraj = sonify.sonify_trajectory_with_sinusoid(multipitch_curve[i], pitch_times, len(audio), Fs=44100)
    fft_sig.append(sineModelSynth(multipitch_curve_amp,multipitch_curve,multipitch_curve_phase))
    axarr[0].plot(pitch_times, multipitch_curve[i])
axarr[0].set_title('estimated multipitch [Hz]')

plt.show()
plt.savefig("singlepitch.png")
