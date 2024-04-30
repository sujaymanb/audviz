import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from pydub import AudioSegment
import simpleaudio as sa
from scipy.signal import savgol_filter

def load_stereo_audio(filename):
    audio = AudioSegment.from_file(filename)
    if audio.channels != 2:
        raise ValueError("Audio file is not stereo.")
    samples = np.array(audio.get_array_of_samples())
    left_channel = samples[0::2]
    right_channel = samples[1::2]
    return left_channel, right_channel, audio.frame_rate

# Load the audio file
left_channel, right_channel, frame_rate = load_stereo_audio("raindrops.wav")
chunk_size = 1024  # Define chunk size

#left_channel = smooth(left_channel)
#right_channel = smooth(right_channel)

# Play the audio
play_obj = sa.play_buffer(np.column_stack((left_channel, right_channel)).tobytes(), num_channels=2, bytes_per_sample=2, sample_rate=frame_rate)

# Set up the plot for animation
fig, ax = plt.subplots()
norm = Normalize(vmin=50, vmax=150)  # Adjust vmin and vmax appropriately

# Old vals to fade gradually
global old_w,old_freq,old_mag_left,old_mag_right
old_w = np.array([])
old_freq = np.array([])
old_mag_left = np.array([])
old_mag_right = np.array([])

decay = 0.99
max_old_vals = 4096

# axes for current vals
sc_left = ax.scatter([], [], c=[], marker='s', cmap='Greens', s=35, norm=norm, alpha=0.7)
sc_right = ax.scatter([], [], c=[], marker='s', cmap='Greens', s=35, norm=norm, alpha=0.7)

# axes for old vals
sc_old_left = ax.scatter([], [], c=[], marker='s', cmap='Greens', s=35, norm=norm, alpha=0.7)
sc_old_right = ax.scatter([], [], c=[], marker='s', cmap='Greens', s=35, norm=norm, alpha=0.7)

# axes format
plt.colorbar(sc_left, label='Magnitude (Log Scale)')
ax.set_xlim(-1, 1)  # Panning from left (-1) to right (1)
ax.set_ylim(0, frame_rate // 2)  # Frequency range from 0 to Nyquist frequency
ax.set_xlabel('Stereo Position')
ax.set_ylabel('Frequency (Hz)')
ax.set_title('Real-time Mid-Side Stereo Image Analysis')


def init():
    sc_left.set_offsets(np.c_[[], []])
    sc_left.set_array(np.array([]))
    
    sc_right.set_offsets(np.c_[[], []])
    sc_right.set_array(np.array([]))

    sc_old_left.set_offsets(np.c_[[], []])
    sc_old_left.set_array(np.array([]))

    sc_old_right.set_offsets(np.c_[[], []])
    sc_old_right.set_array(np.array([]))

    return sc_left, sc_right

def in_decibels(magnitude):
    return 20 * np.log10(magnitude + 1e-10)

def calculate_stereo_width(left_fft, right_fft):
    # Simple difference measure, other methods could be used
    width = np.abs(left_fft - right_fft) / (np.abs(left_fft) + np.abs(right_fft) + 1e-10)  # Avoid division by zero
    return width

def append_max(a, b, max_l):
    a = np.append(a,b)
    #print(f"old sh: {a.shape}")
    if (a.shape[0] > max_l):
        a = a[-max_l:]

    return a

def add_to_old_buff(stereo_width,frequencies,magnitude_left,magnitude_right):
    global old_w,old_freq,old_mag_left,old_mag_right
    old_w = append_max(old_w, stereo_width, max_old_vals)
    old_freq = append_max(old_freq, frequencies, max_old_vals)
    old_mag_left = append_max(old_mag_left, magnitude_left, max_old_vals)
    old_mag_right = append_max(old_mag_right, magnitude_right, max_old_vals)

def apply_smoothing(vals):
    vals = savgol_filter(vals, 100, 3, axis=0)
    return vals


def update_old_vals():
    global old_mag_left, old_mag_right, old_w, old_freq
    # decay old vals
    old_mag_left = decay * old_mag_left + 1e-10
    old_mag_right = decay * old_mag_right + 1e-10

    # update left old vals
    smooth_w = apply_smoothing(old_w)
    #get_histogram(old_w, old_freq, old_mag_left)

    sc_old_left.set_offsets(np.c_[-smooth_w, old_freq])
    sc_old_left.set_array(np.array(old_mag_left))

    # update right old vals
    sc_old_right.set_offsets(np.c_[smooth_w, old_freq])
    sc_old_right.set_array(np.array(old_mag_right))

    print(f"old mag l: {old_mag_left}")
    print(f"old mag r: {old_mag_right}")

def update(frame):
    start_index = frame * chunk_size
    end_index = start_index + chunk_size
    
    if end_index > len(left_channel):
        return sc_left, sc_right
    
    left_chunk = left_channel[start_index:end_index]
    right_chunk = right_channel[start_index:end_index]
    
    left_fft = np.fft.rfft(left_chunk)
    right_fft = np.fft.rfft(right_chunk)
    
    frequencies = np.linspace(0, frame_rate // 2, len(left_fft))

    magnitude_left = in_decibels(np.abs(left_fft))
    magnitude_right = in_decibels(np.abs(right_fft))
    
    stereo_width = calculate_stereo_width(left_fft, right_fft)
    
    print(f"shape: {magnitude_left.shape}")
    print(f"Updating left {frame}: {magnitude_left[:5]}")
    print(f"Updating right {frame}: {magnitude_right[:5]}")
    
    add_to_old_buff(stereo_width,frequencies,magnitude_left,magnitude_right)
    update_old_vals()

    sc_left.set_offsets(np.c_[-stereo_width, frequencies])  # Mirroring left channel
    sc_left.set_array(magnitude_left)
    
    sc_right.set_offsets(np.c_[stereo_width, frequencies])
    sc_right.set_array(magnitude_right)
    
    return sc_left, sc_right

# Animation
ani = FuncAnimation(fig, update, init_func=init, frames=range(len(left_channel) // chunk_size), interval=(chunk_size / frame_rate) * 1000, blit=True)

plt.show()
play_obj.wait_done()