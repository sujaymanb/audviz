import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from pydub import AudioSegment
import simpleaudio as sa
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic_2d
from scipy.ndimage import maximum_filter, gaussian_filter

# params
decay = 0.95
imsize = 15
chunk_size = 1024  # Define chunk size
past_buffer = []

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

# Play the audio
play_obj = sa.play_buffer(np.column_stack((left_channel, right_channel)).tobytes(), num_channels=2, bytes_per_sample=2, sample_rate=frame_rate)

# Set up the plot for animation
fig, ax = plt.subplots()
#norm = Normalize(vmin=50, vmax=150)  # Adjust vmin and vmax appropriately

# init lr images
global left_img,right_img

left_img = np.zeros((imsize,imsize))
right_img = np.zeros((imsize,imsize))

# initialize plot
fig, ax = plt.subplots()
comb_img = ax.imshow(np.zeros((imsize,2*imsize)),cmap='pink')
comb_img.set_clim(vmin=50, vmax=140)

# set display settings
ax.set_yticklabels([])
ax.set_xticklabels([])
#ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)

# axes labels and limits
plt.colorbar(comb_img, label='Magnitude (Log Scale)')
#ax.set_xlim(-1, 1)  # Panning from left (-1) to right (1)
#ax.set_ylim(0, frame_rate // 2)  # Frequency range from 0 to Nyquist frequency
ax.set_xlabel('Stereo Position')
ax.set_ylabel('Frequency (Hz)')
ax.set_title('Real-time Mid-Side Stereo Image Analysis')

# start imshow with initial combined image
def init():
    comb_img.set_data(np.zeros((imsize,2*imsize)))

    return [comb_img]

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

def apply_smoothing(vals,win_size=20):
    if len(past_buffer) < win_size:
        return vals

    new_vals = savgol_filter(np.array(past_buffer), win_size, 3, axis=0)
    return new_vals

def fill_nan(data):
    # Fill in NaN's by interpolating
    mask = np.isnan(data)
    data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
    return data

def get_grid(widths,freqs,mags,flip=False):
    # for x, bin widths then compute the mean mag for the bins
    binned_mags,_,_,_ = binned_statistic_2d(widths,freqs,mags,bins=imsize)

    #binned_mags = fill_nan(binned_mags)
    binned_mags = np.nan_to_num(binned_mags)

    if flip:
        binned_mags = np.flip(binned_mags,axis=1)

    return binned_mags

def process_frame(frame):
    start_index = frame * chunk_size
    end_index = start_index + chunk_size
    
    if end_index > len(left_channel):
        return None
    
    left_chunk = left_channel[start_index:end_index]
    right_chunk = right_channel[start_index:end_index]
    
    left_fft = np.fft.rfft(left_chunk)
    right_fft = np.fft.rfft(right_chunk)
    
    frequencies = np.linspace(0, frame_rate // 2, len(left_fft))

    magnitude_left = in_decibels(np.abs(left_fft))
    magnitude_right = in_decibels(np.abs(right_fft))
    
    stereo_width = calculate_stereo_width(left_fft, right_fft)

    return stereo_width, frequencies, magnitude_left, magnitude_right


def update(frame):
    global left_img, right_img
    ret = process_frame(frame)
    if ret is None:
        return [comb_img]
    
    (w,f,ml,mr) = ret

    #print(f"shape: {ml.shape} {mr.shape}")
    #print(f"Updating left {frame}: {ml[:5]}")
    #print(f"Updating right {frame}: {mr[:5]}")
    
    # get grids
    # flip left side so mono is in middle
    new_left_img = get_grid(w,f,ml,flip=True)
    new_right_img = get_grid(w,f,mr)
    #print(f"l shape: {left_img.shape}")

    # apply decay so previous values fade slowly
    #left_img = (0.5 * decay * left_img) + (0.5 * new_left_img)
    #right_img = (0.5 * decay * right_img) + (0.5 * new_right_img)

    left_img = new_left_img
    right_img = new_right_img

    # concatenate into full left right image
    full_img = np.concatenate((left_img,right_img), axis=1)
    past_buffer.append(full_img)
    print(f"min: {np.min(full_img)} max: {np.max(full_img)}")

    # smooth image
    #full_img = maximum_filter(full_img, 2)
    #full_img = gaussian_filter(full_img, 3)

    # set the data for imshow
    comb_img.set_data(full_img)
    
    # must return as iterable collection of artists
    return [comb_img]

# Animation
ani = FuncAnimation(fig, update, init_func=init, frames=range(len(left_channel) // chunk_size), interval=(chunk_size / frame_rate) * 1000, blit=True)

plt.show()
play_obj.wait_done()