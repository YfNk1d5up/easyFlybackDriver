import essentia
import essentia.standard as ess
import numpy as np
import tqdm


# input and output files
inputFilename = 'data/input/takeonme/other.wav'
outputFilename = 'data/output/takeonme/other_multi.wav'
# algorithm parameters
params = {
    "frameSize": 2048,
    "hopSize": 512,
    "startFromZero": False,
    "sampleRate": 44100,
    "maxnSines": 2,
    "minSineDur": 0.02,
    "freqDevOffset": 10,
    "freqDevSlope": 0.001,
    "minFreq":300,
    "maxFreq":1250,
    "magnitudeThreshold":30
}

# initialize some algorithms
loader = ess.MonoLoader(
    filename=str(inputFilename), sampleRate=params["sampleRate"]
)
multipitchExtractor = ess.MultiPitchMelodia(
    frameSize=params["frameSize"],
    hopSize=params["hopSize"],
    sampleRate=params["sampleRate"],
    minFrequency=params["minFreq"],
    maxFrequency=params["maxFreq"],
    magnitudeThreshold= params["magnitudeThreshold"]
)
predominantPitchExtractor = ess.PredominantPitchMelodia(
    frameSize=params["frameSize"],
    hopSize=params["hopSize"],
    sampleRate=params["sampleRate"]
)
keyExtractor = ess.KeyExtractor()
rhythmExtractor = ess.RhythmExtractor()
sineModelAnal = ess.SineModelAnal()
filter = ess.BandPass(
    bandwidth=params["maxFreq"]-params["minFreq"],
    cutoffFrequency=params["maxFreq"],
    sampleRate=params["sampleRate"]
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
duration = len(audio) / params["sampleRate"]
mins = duration // 60
secs = duration % 60
print(f'Duration : {int(mins)}m{int(secs)}s')
pitch_curve = multipitchExtractor(audio)

audioout = np.array(0)
# loop over all frames
for freq in tqdm.tqdm(pitch_curve):
    magn = []
    for f in freq:
        magn.append(1)

    outfft = smsynstd(essentia.array(magn),
                   essentia.array(freq),
                   essentia.array([])
                   )

    # STFT synthesis
    out = overlstd(ifftstd(outfft))
    audioout = np.append(audioout, out)


# write audio output
awritestd(audioout.astype(np.float32))