import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import filedialog
import threading
import queue
try:
    import pygame
    pygame_available = True
except ImportError:
    pygame_available = False
    print("Pygame not installed. Audio features will be skipped. Install with: 'pip install pygame'")

# Global variables to store data for sound playback
global_heights = []
global_runs = []
global_freqs = []
audio_queue = queue.Queue()  # Queue to control audio playback
audio_thread = None
stop_audio_flag = False

def load_binary_data():
    """Load binary data from a file selected by the user."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title="Select a file to sonify")
    if not file_path:
        print("No file selected. Exiting.")
        exit(1)

    bit_limit = 5000  # Process 5,000 bits
    bit_offset = 10000  # Skip the first 10,000 bits
    binary = ""
    try:
        with open(file_path, "rb") as file:
            file.seek(bit_offset // 8)  # Skip to the desired byte offset
            byte_data = file.read(bit_limit // 8 + 1)  # Read enough bytes
            for byte in byte_data:
                binary += format(byte, '08b')  # Convert each byte to 8-bit binary string
        binary = binary[:bit_limit]  # Trim to desired bit length
    except FileNotFoundError:
        print(f"File not found: {file_path}.")
        exit(1)
    return binary, file_path

def process_binary_data(binary):
    """Process the binary string to find runs and calculate heights."""
    runs = []
    current_run = binary[0]
    current_count = 1
    i = 1
    while i < len(binary):
        if i < len(binary) - 1 and binary[i] != binary[i-1] and binary[i] != binary[i+1]:
            start_idx = i - 1
            alternating_run = binary[start_idx:i+1]
            j = i + 1
            while j < len(binary) and binary[j] != binary[j-1]:
                alternating_run += binary[j]
                j += 1
            runs.append(('alternating', len(alternating_run)))
            i = j
        else:
            if binary[i] == current_run[-1]:
                current_count += 1
            else:
                runs.append((current_run, current_count))
                current_run = binary[i]
                current_count = 1
            i += 1
    if current_count > 0:
        runs.append((current_run, current_count))

    heights = [0]  # Starting height
    bit_idx = 0
    for run_type, run_length in runs:
        if run_type == 'alternating':
            for _ in range(run_length):
                heights.append(heights[-1])
            bit_idx += run_length
        else:
            step = run_length if run_type == '1' else -run_length
            for i in range(run_length):
                heights.append(heights[-1] + step / run_length)
            bit_idx += run_length

    heights = heights[:len(binary) + 1]
    if len(heights) != len(binary) + 1:
        print(f"Warning: Height length mismatch. Expected {len(binary) + 1}, got {len(heights)}")

    return runs, heights

def compute_rate_of_change(heights):
    """Compute the rate of change of heights and smooth it."""
    rate_of_change = np.diff(heights)  # Length: 5000
    # Smooth with a larger moving average for sound
    window_size = 200
    smoothed_rate = np.convolve(rate_of_change, np.ones(window_size)/window_size, mode='valid')
    # Pad to match length of heights[1:] (5000)
    pad_before = (window_size - 1) // 2
    pad_after = (window_size - 1) - pad_before
    smoothed_rate = np.pad(smoothed_rate, (pad_before, pad_after), mode='edge')
    # Ensure length matches
    if len(smoothed_rate) < len(rate_of_change):
        smoothed_rate = np.pad(smoothed_rate, (0, len(rate_of_change) - len(smoothed_rate)), mode='edge')
    elif len(smoothed_rate) > len(rate_of_change):
        smoothed_rate = smoothed_rate[:len(rate_of_change)]

    # Smooth for display
    display_window_size = 50
    display_smoothed_rate = np.convolve(rate_of_change, np.ones(display_window_size)/display_window_size, mode='valid')
    pad_before_display = (display_window_size - 1) // 2
    pad_after_display = (display_window_size - 1) - pad_before_display
    display_smoothed_rate = np.pad(display_smoothed_rate, (pad_before_display, pad_after_display), mode='edge')
    if len(display_smoothed_rate) < len(rate_of_change):
        display_smoothed_rate = np.pad(display_smoothed_rate, (0, len(rate_of_change) - len(display_smoothed_rate)), mode='edge')
    elif len(display_smoothed_rate) > len(rate_of_change):
        display_smoothed_rate = display_smoothed_rate[:len(rate_of_change)]

    return rate_of_change, smoothed_rate, display_smoothed_rate

def map_to_pentatonic(smoothed_rate):
    """Map the smoothed rate of change to a pentatonic scale for sound."""
    base_freqs = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25]
    num_notes = len(base_freqs)
    rate_min, rate_max = min(smoothed_rate), max(smoothed_rate)
    rate_range = rate_max - rate_min
    if rate_range != 0:
        normalized_rates = [(r - rate_min) / rate_range for r in smoothed_rate]
        freq_indices = [int(n * (num_notes - 1)) for n in normalized_rates]
        freqs = [base_freqs[i] for i in freq_indices]
    else:
        freqs = [base_freqs[0]] * len(smoothed_rate)
    return freqs

def play_sound_thread():
    """Play sound in a separate thread."""
    pygame.init()
    pygame.mixer.init()
    print("Playing sound... (click 'Stop Sound' to stop)")
    segment_size = 200
    sound_idx = 0
    while sound_idx < len(global_freqs):
        if stop_audio_flag:
            break
        end_idx = min(sound_idx + segment_size, len(global_freqs))
        freq = global_freqs[sound_idx]
        sample_rate = 44100
        duration = 0.1 * (end_idx - sound_idx) / 50
        t = np.linspace(0, duration, int(sample_rate * duration))
        sound = 0.5 * np.sin(2 * np.pi * freq * t)
        sound = (sound * 32767).astype(np.int16)
        sound = np.stack([sound, sound], axis=1)
        pygame.mixer.Sound(sound).play()
        pygame.time.wait(int(duration * 1000))
        sound_idx = end_idx
    pygame.quit()
    print("Sound stopped.")

def play_sound(event):
    """Start the sound playback in a separate thread."""
    global audio_thread, stop_audio_flag
    if not pygame_available:
        print("Pygame not available. Cannot play sound.")
        return
    stop_audio_flag = False
    audio_thread = threading.Thread(target=play_sound_thread)
    audio_thread.start()

def stop_sound(event):
    """Stop the sound playback."""
    global stop_audio_flag
    stop_audio_flag = True
    if audio_thread is not None:
        audio_thread.join()  # Wait for the thread to finish

def main():
    """Main function to load data, process it, and display the graph with interactive buttons."""
    binary, file_path = load_binary_data()
    runs, heights = process_binary_data(binary)
    rate_of_change, smoothed_rate, display_smoothed_rate = compute_rate_of_change(heights)
    freqs = map_to_pentatonic(smoothed_rate)

    global global_heights, global_runs, global_freqs
    global_heights = heights
    global_runs = runs
    global_freqs = freqs

    x = list(range(len(binary)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    bit_idx = 0
    for run_type, run_length in runs:
        x_segment = x[bit_idx:bit_idx + run_length]
        y_segment = heights[bit_idx + 1:bit_idx + run_length + 1]
        if run_type == 'alternating':
            ax1.plot(x_segment, y_segment, color='red', linestyle='-', marker='o', markersize=2, label='Alternating Runs' if bit_idx == 0 else "")
        else:
            ax1.plot(x_segment, y_segment, color='blue', linestyle='-', marker='o', markersize=2, label='Height Path' if bit_idx == 0 else "")
        bit_idx += run_length

    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax1.set_title(f"{file_path.split('/')[-1]} as Binary Step Graph (Inspired by Nintendo Sound Chip)")
    ax1.set_ylabel("Height (Y)")
    ax1.grid(True, linestyle='--', alpha=0.7)

    min_height = min(heights[1:])
    max_height = max(heights[1:])
    min_idx = heights[1:].index(min_height)
    max_idx = heights[1:].index(max_height)
    ax1.scatter([min_idx], [min_height], color='red', label=f'Min: {min_height}', zorder=5)
    ax1.scatter([max_idx], [max_height], color='green', label=f'Max: {max_height}', zorder=5)

    window_size = 10
    moving_avg = np.convolve(heights[1:], np.ones(window_size)/window_size, mode='valid')
    ma_x = np.arange(window_size - 1, len(heights[1:]))
    ma_x = ma_x[:len(moving_avg)]
    ax1.plot(ma_x, moving_avg, color='orange', linestyle='--', label=f'Moving Avg (window={window_size})')

    ax2.plot(x, rate_of_change, color='gray', alpha=0.5, label='Rate of Change', linewidth=0.5)
    ax2.plot(x, display_smoothed_rate, color='purple', label='Smoothed Rate of Change', linewidth=1.5)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax2.set_xlabel("Bit Iteration (X)")
    ax2.set_ylabel("Rate of Change")
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Add play and stop buttons
    ax_play = plt.axes([0.71, 0.01, 0.1, 0.05])
    btn_play = Button(ax_play, 'Play Sound')
    btn_play.on_clicked(play_sound)

    ax_stop = plt.axes([0.81, 0.01, 0.1, 0.05])
    btn_stop = Button(ax_stop, 'Stop Sound')
    btn_stop.on_clicked(stop_sound)

    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.show()

    print(f"Total bits processed: {len(binary)}")
    print(f"Total runs: {len(runs)}")
    print(f"Final height: {heights[-1]}")
    print(f"Minimum height: {min_height}")
    print(f"Maximum height: {max_height}")

if __name__ == "__main__":
    main()