import fftProcess
import synthetize_notes
import argparse

"Polyphia-G.O.A.T.-_Official-Music-Video"
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="file to process (wav format)", default="test-Polyphia-G.O.A.T")
    args = parser.parse_args()
    FILE_NAME = str(args.file)
    fftProcess.getTopNotesCSV(FILE_NAME, FFT_WINDOW_SECONDS=0.05,FREQ_MIN=300, FREQ_MAX=1250, TOP_NOTES=3)
    synthetize_notes.save_wav_multi(FILE_NAME, TOP_NOTES=3, sampling_rate=16000, FFT_WINDOW_SECONDS=0.05, amplitude=1000)