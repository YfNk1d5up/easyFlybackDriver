# import essentia in standard mode
import essentia
import essentia.standard as es
import tqdm

# We'll need to some numerical tools and to define filepaths
import numpy as np
from pathlib import Path

# input and output files
#tutorial_dir = Path(__file__).resolve().parent
inputFilename = 'streaming/other.wav'
outputFilename = 'harmonic_sub_synthesis.wav'

params = {
    "frameSize": 2048,
    "hopSize": 1024,
    "startFromZero": False,
    "sampleRate": 44100.0,
    "maxnSines": 4,
    "magnitudeThreshold": -47,
    "minSineDur": 0.02,
    "freqDevOffset": 10,
    "freqDevSlope": 0.001,
    "attenuation_dB":100,
    "maskbinwidth":2
}

# create an audio loader and import audio file
audio = es.MonoLoader(filename=str(inputFilename),
                      sampleRate=params["sampleRate"])()
print(f"Duration of the audio sample [sec]: {len(audio) / params['sampleRate']:.3f}")


# initialize some algorithms
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

hmask = es.HarmonicMask(
    sampleRate=params["sampleRate"],
    binWidth=params["maskbinwidth"],
    attenuation=params["attenuation_dB"]
)



# init output audio array
audioout = np.array(0)

# loop over all frames
for idx, frame in tqdm.tqdm(enumerate(
    es.FrameGenerator(audio, frameSize=params["frameSize"],
                      hopSize=params["hopSize"]
                      )
)):

    # STFT analysis
    fft0 = fft(w(frame))
    frequencies, magnitudes, phases = smanal(fft0)
    # get pitch of current frame
    m0 = magnitudes[0]
    f0 = frequencies[0]
    p0 = phases[0]
    # here we  apply the harmonic mask spectral transformations
    fft1 = hmask(fft0, f0)
    frequencies, magnitudes, phases = smanal(fft1)
    # get pitch of current frame
    m1 = magnitudes[0]
    f1 = frequencies[0]
    p1 = phases[0]
    fft2 = hmask(fft1, f1)
    frequencies, magnitudes, phases = smanal(fft2)
    # get pitch of current frame
    m2 = magnitudes[0]
    f2 = frequencies[0]
    p2 = phases[0]
    fft3 = hmask(fft2, f2)
    frequencies, magnitudes, phases = smanal(fft3)
    # get pitch of current frame
    m3 = magnitudes[0]
    f3 = frequencies[0]
    p3 = phases[0]
    m = [m0, m1, m2, m3]
    f = [f0, f1, f2, f3]
    p = [p0, p1, p2, p3]
    # STFT synthesis
    outfft = smsyn(essentia.array(m),
                      essentia.array(f),
                      essentia.array(p)
                      )
    # STFT synthesis
    out = overl(ifft(outfft))
    audioout = np.append(audioout, out)

# write audio output
awrite(audioout.astype(np.float32))