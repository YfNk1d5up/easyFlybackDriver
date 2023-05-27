import essentia
import essentia.streaming as es
import essentia.standard as ess
import numpy as np
import tqdm
from pathlib import Path

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

# input and output files
inputFilename = 'other.wav'
outputFilename = 'synthesis_sinemodel.wav'

# initialize some algorithms
loader = es.MonoLoader(
    filename=str(inputFilename), sampleRate=params["sampleRate"]
)
filter = es.BandPass(
    bandwidth=700,
    cutoffFrequency=880,
    sampleRate=params["sampleRate"]
)
fcut = es.FrameCutter(
    frameSize=params["frameSize"],
    hopSize=params["hopSize"],
    startFromZero=False,
)
w = es.Windowing(type="blackmanharris92")
fft = es.FFT(size=params["frameSize"])
smanal = es.SineModelAnal(
    sampleRate=params["sampleRate"],
    maxnSines=params["maxnSines"],
    magnitudeThreshold=params["magnitudeThreshold"],
    freqDevOffset=params["freqDevOffset"],
    freqDevSlope=params["freqDevSlope"],
)
pExt = es.PredominantPitchMelodia(
    frameSize=params["frameSize"],
    hopSize=params["hopSize"],
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
one = es.UnaryOperator(scale=0,
                       shift=1000,
                       type="identity"
                       )
zero = es.UnaryOperator(scale=0,
                       shift=0,
                       type="identity"
                       )
pool = essentia.Pool()


# Define a network of connected algorithms

# analysis
loader.audio >> filter.signal
filter.signal >> pExt.signal


pExt.pitch >> (pool, "frequencies")
pExt.pitchConfidence >> None

# run the network
essentia.run(loader)


freqs = pool["frequencies"].flatten()
# init output audio array
audioout = np.array(0)
# loop over all frames
for freq in tqdm.tqdm(freqs):

    outfft = smsynstd(np.array([1]),
                   np.array([freq]),
                   np.array([0])
                   )

    # STFT synthesis
    out = overlstd(ifftstd(outfft))
    audioout = np.append(audioout, out)


# write audio output
awritestd(audioout.astype(np.float32))
essentia.reset(loader)