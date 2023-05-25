import essentia
import essentia.streaming as es
from pathlib import Path

# algorithm parameters
params = {
    "frameSize": 2048,
    "hopSize": 512,
    "startFromZero": False,
    "sampleRate": 44100,
    "maxnSines": 4,
    "magnitudeThreshold": 40,
    "minSineDur": 0.02,
    "freqDevOffset": 10,
    "freqDevSlope": 0.001,
}

# input and output files

inputFilename = 'other.wav'
outputFilename = 'synthesis.wav'

# initialize some algorithms
loader = es.MonoLoader(
    filename=str(inputFilename), sampleRate=params["sampleRate"]
)
fcut = es.FrameCutter(
    frameSize=params["frameSize"],
    hopSize=params["hopSize"],
    startFromZero=False,
)
w = es.Windowing(type="blackmanharris92")
fft = es.FFT(size=params["frameSize"])
"""
multipitchExtractor = es.MultiPitchMelodia(
    binResolution=10,
    filterIterations=3,
    frameSize=params["frameSize"],
    guessUnvoiced=False,
    harmonicWeight=0.8,
    hopSize=params["hopSize"],
    magnitudeCompression=1,
    magnitudeThreshold=params["magnitudeThreshold"],
    maxFrequency=20000,
    minDuration=100,
    minFrequency=80,
    numberHarmonics=20,
    peakDistributionThreshold=0.9,
    peakFrameThreshold=0.9,
    pitchContinuity=27.5625,
    referenceFrequency=55,
    sampleRate=params["sampleRate"],
    timeContinuity=100
)
"""
multipitchExtractor = es.MultiPitchMelodia(
    frameSize=params["frameSize"],
    hopSize=params["hopSize"],
    magnitudeThreshold=params["magnitudeThreshold"],
    sampleRate=params["sampleRate"]
)
smanal = es.SineModelAnal(
    sampleRate=params["sampleRate"],
    maxnSines=params["maxnSines"],
    magnitudeThreshold=params["magnitudeThreshold"],
    freqDevOffset=params["freqDevOffset"],
    freqDevSlope=params["freqDevSlope"],
)
smsyn = es.SineModelSynth(
    sampleRate=params["sampleRate"],
    fftSize=params["frameSize"],
    hopSize=params["hopSize"],
)
ifft = es.IFFT(size=params["frameSize"])
overl = es.OverlapAdd(frameSize=params["frameSize"], hopSize=params["hopSize"])
awrite = es.MonoWriter(
    filename=str(outputFilename), sampleRate=params["sampleRate"]
)
pool = essentia.Pool()


# Define a network of connected algorithms

# analysis
loader.audio >> fcut.signal
fcut.frame >> w.frame
w.frame >> fft.frame
fft.fft >> smanal.fft
smanal.magnitudes >> (pool, "magnitudes")
smanal.frequencies >> (pool, "frequencies")
smanal.phases >> (pool, "phases")

# synthesis
smanal.magnitudes >> smsyn.magnitudes
smanal.frequencies >> smsyn.frequencies
smanal.phases >> smsyn.phases
smsyn.fft >> ifft.fft
ifft.frame >> overl.frame
overl.signal >> awrite.audio
overl.signal >> (pool, "audio")

# run the network
essentia.run(loader)