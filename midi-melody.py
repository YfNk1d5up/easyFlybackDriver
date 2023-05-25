from basic_pitch.inference import predict_and_save, predict


predict_and_save(
    audio_path_list=["separated/htdemucs/test-Polyphia-G.O.A.T/bass.wav"],
    output_directory="data",
    save_midi=True,
    sonify_midi=True,
    save_model_outputs=False,
    save_notes=False,
    onset_threshold=0.5,
    frame_threshold=0.3,
    minimum_note_length=58,
    minimum_frequency=50,
    maximum_frequency=1000,
    multiple_pitch_bends=False,
    debug_file=None,
    sonification_samplerate=44100,
    midi_tempo=107
)
"""

model_output, midi_data, note_events = predict(
    audio_path="separated/htdemucs/test-Polyphia-G.O.A.T/other.wav",
    onset_threshold=0.5,
    frame_threshold=0.22,
    minimum_note_length=58,
    minimum_frequency=250,
    maximum_frequency=1440,
    multiple_pitch_bends=False,
    debug_file=None,
    midi_tempo=107
)
print(note_events)
"""