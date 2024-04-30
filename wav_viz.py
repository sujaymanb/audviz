import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pydub import AudioSegment
import simpleaudio as sa

# Load your WAV file
audio = AudioSegment.from_file("raindrops.wav")
# Assuming the audio is stereo, extract channels
samples = np.array(audio.get_array_of_samples())

if audio.channels == 2:
    right_channel = samples[1::2]  # Take every second sample starting from index 1
else:
    right_channel = samples  # Fallback to mono if not stereo

# Parameters
frame_rate = audio.frame_rate
chunk_size = 1024  # Adjust the chunk size depending on the file's sample rate

# Calculate duration of one chunk in milliseconds
chunk_duration = (chunk_size / frame_rate) * 1000  # Duration in milliseconds

# Play the audio
play_obj = sa.play_buffer(right_channel.tobytes(), num_channels=1, bytes_per_sample=audio.sample_width, sample_rate=audio.frame_rate)

# Set up the plot
fig, ax = plt.subplots()
xf = np.linspace(0, frame_rate // 2, chunk_size // 2 + 1)
line, = ax.plot(xf, np.zeros(chunk_size // 2 + 1))
ax.set_yscale('log')  # Set y-axis to logarithmic scale
ax.set_ylim(0, 10**6)  # Set the y-axis limits appropriately

# Update function for the animation with logarithmic magnitude
def update(frame):
    start_index = frame * chunk_size
    end_index = start_index + chunk_size
    if end_index > len(right_channel):
        print("End of audio")
        return line,
    chunk = right_channel[start_index:end_index]
    yf = np.fft.rfft(chunk)
    yf = np.abs(yf)
    #yf = 20 * np.log10(yf + 1)  # Logarithmic scale, adding 1 to avoid log(0)
    line.set_ydata(yf)
    return line,

# Animation
ani = FuncAnimation(fig, update, frames=range(len(right_channel) // chunk_size), interval=chunk_duration, blit=True)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Real-time Spectrum Analysis of Right Channel')
plt.show()

# Wait for playback to finish
play_obj.wait_done()
