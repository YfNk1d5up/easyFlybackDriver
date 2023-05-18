import csv
import numpy as np
import os
import wave

sampling_rate = 16000  # Sampling rate (Hz)
TOP_NOTES = 5
CSV_FILE_NAME = "take_on_me_8bit"

columns = ["time"]
for n in range(TOP_NOTES):
    columns.append(f"note{n}")
    columns.append(f"anote{n}")
# Read CSV file
with open(f"data/output/{CSV_FILE_NAME}.csv") as csv_file:
    csv_reader = list(csv.reader(csv_file, delimiter=','))
    notes = []
    for row in csv_reader[1:]:
        dict_note = {}
        for i in range(len(columns)):
            dict_note[columns[i]] = row[i]
        notes.append(dict_note)

FFT_WINDOW_SECONDS = 0.05
amplitude = 10000  # Amplitude of the wave (arbitrary units)

samples = []
samplevoice = [[] for n in range(TOP_NOTES)]
for note in notes:
    time = np.linspace(float(note['time']), float(note['time']) + FFT_WINDOW_SECONDS, int(FFT_WINDOW_SECONDS * sampling_rate), False)
    freqs = []
    amps = []
    for i in range(1,len(columns),2):
        try:
            freqs.append(float(note[columns[i]]))
            amps.append(float(note[columns[i+1]]))
        except:
            freqs.append(0)
            amps.append(1)
    sine_wave = 0
    sine_wave_voices = [0, 0, 0, 0]
    for i in range(len(freqs)):
        sine_wave+=amps[i] * np.sin(2 * np.pi * freqs[i] * time)
        samplevoice[i].extend(amplitude * np.sin(2 * np.pi * freqs[i] * time))
    sine_wave*=amplitude/amps[0]
    samples.extend(sine_wave)

os.mkdir(f"data/output/{CSV_FILE_NAME}")

wave_file = wave.open(f"data/output/{CSV_FILE_NAME}/4voices_synthetize.wav", 'wb')
wave_file.setnchannels(1)
wave_file.setsampwidth(2)
wave_file.setframerate(sampling_rate)
wave_file.writeframes(np.array(samples).astype(np.int16).tobytes())
wave_file.close()

for i in range(len(samplevoice)):
    wave_file = wave.open(f"data/output/{CSV_FILE_NAME}/4voices_synthetize{i}.wav", 'wb')
    wave_file.setnchannels(1)
    wave_file.setsampwidth(2)
    wave_file.setframerate(sampling_rate)
    wave_file.writeframes(np.array(samplevoice[i]).astype(np.int16).tobytes())
    wave_file.close()