from pydub import AudioSegment
import numpy as np

def in_decibels(magnitude):
    return 20 * np.log10(magnitude + 1e-10)

def load_and_inspect_audio(filename):
    audio = AudioSegment.from_file(filename)
    if audio.channels != 2:
        raise ValueError("Audio file is not stereo.")
    samples = np.array(audio.get_array_of_samples())
    left_channel = samples[0::2]
    right_channel = samples[1::2]
    left_fft = np.fft.rfft(left_channel)
    right_fft = np.fft.rfft(right_channel)
    magnitude_left = in_decibels(np.abs(left_fft))
    magnitude_right = in_decibels(np.abs(right_fft))
    print("Sample magnitudes (left):", magnitude_left[:10])
    print("Sample magnitudes (right):", magnitude_right[:10])

load_and_inspect_audio("raindrops.wav")