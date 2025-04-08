import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons
import tkinter as tk
from tkinter import filedialog
import threading
import queue
import os
try:
    import pygame
    import pygame._sdl2.audio as sdl2_audio
    pygame_available = True
except ImportError:
    pygame_available = False
    print("Pygame not installed. Audio features will be skipped. Install with: 'pip install pygame'")

# Global variables to store data for sound playback.
global_heights = []
global_runs = []
global_freqs = []
global_raw_heights = []
audio_queue = queue.Queue()
audio_thread = None
stop_audio_flag = False
playback_mode = "Smooth Transitions"

def get_audio_devices():
    """Get a list of available audio output devices."""
    try:
        init_by_me = not pygame.mixer.get_init()
        if init_by_me:
            pygame.mixer.init()
        devices = tuple(sdl2_audio.get_audio_device_names(False))
        if init_by_me:
            pygame.mixer.quit()
        return devices
    except Exception as e:
        print(f"Error retrieving audio devices: {e}")
        return []

def load_binary_data():
    """Load binary data from a file selected by the user."""
    try:
        if getattr(sys, 'frozen', False):
            os.chdir(sys._MEIPASS)
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Select a file to sonify")
        if not file_path:
            print("No file selected. Exiting.")
            exit(1)

        bit_limit = 5000
        bit_offset = 10000
        binary = ""
        with open(file_path, "rb") as file:
            file.seek(bit_offset // 8)
            byte_data = file.read(bit_limit // 8 + 1)
            if len(byte_data) < (bit_limit // 8):
                raise ValueError("File is too small or empty after the offset.")
            for byte in byte_data:
                binary += format(byte, '08b')
        binary = binary[:bit_limit]
        if not binary:
            raise ValueError("No binary data extracted from the file.")
        return binary, file_path
    except FileNotFoundError:
        print(f"Error: File not found. Please select a valid file.")
        exit(1)
    except ValueError as ve:
        print(f"Error: {ve}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error while loading file: {e}")
        exit(1)

def process_binary_data(binary):
    """Process the binary string to find runs and calculate heights."""
    try:
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

        heights = [0]
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
    except Exception as e:
        print(f"Error processing binary data: {e}")
        exit(1)

def compute_rate_of_change(heights):
    """Compute the rate of change of heights and smooth it."""
    try:
        rate_of_change = np.diff(heights)
        window_size = 200
        smoothed_rate = np.convolve(rate_of_change, np.ones(window_size)/window_size, mode='valid')
        pad_before = (window_size - 1) // 2
        pad_after = (window_size - 1) - pad_before
        smoothed_rate = np.pad(smoothed_rate, (pad_before, pad_after), mode='edge')
        if len(smoothed_rate) < len(rate_of_change):
            smoothed_rate = np.pad(smoothed_rate, (0, len(rate_of_change) - len(smoothed_rate)), mode='edge')
        elif len(smoothed_rate) > len(rate_of_change):
            smoothed_rate = smoothed_rate[:len(rate_of_change)]

        display_window_size = 50
        display_smoothed_rate = np.convolve(rate_of_change, np.ones(display_window_size)/display_window_size, mode='valid')
        pad_before_display = (display_window_size - 1) // 2
        pad_after_display = (display_window_size - 1) - pad_before_display
        display_smoothed_rate = np.pad(display_smoothed_rate, (pad_before_display, pad_after_display), mode='edge')
        if len(display_smoothed_rate) < len(rate_of_change):
            display_smoothed_rate = np.pad(display_smoothed_rate, (0, len(rate_of_change) - len(smoothed_rate)), mode='edge')
        elif len(display_smoothed_rate) > len(rate_of_change):
            display_smoothed_rate = display_smoothed_rate[:len(rate_of_change)]

        return rate_of_change, smoothed_rate, display_smoothed_rate
    except Exception as e:
        print(f"Error computing rate of change: {e}")
        exit(1)

def map_to_pentatonic(values, use_derivative=True):
    """Map values (either smoothed rate or raw heights) to a pentatonic scale."""
    try:
        base_freqs = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25]
        num_notes = len(base_freqs)
        if use_derivative:
            val_min, val_max = min(values), max(values)
        else:
            val_min, val_max = min(values[1:]), max(values[1:])
        val_range = val_max - val_min
        if val_range != 0:
            normalized_vals = [(v - val_min) / val_range for v in values]
            freq_indices = [int(n * (num_notes - 1)) for n in normalized_vals]
            freqs = [base_freqs[i] for i in freq_indices]
        else:
            freqs = [base_freqs[0]] * len(values)
        return freqs
    except Exception as e:
        print(f"Error mapping to pentatonic scale: {e}")
        exit(1)

def play_sound_thread():
    """Play sound in a separate thread with user-selected playback mode."""
    try:
        pygame.init()
        devices = get_audio_devices()
        if not devices:
            print("No audio devices found. Using default device.")
            device = None
        else:
            print("Available audio devices:")
            for i, dev in enumerate(devices):
                print(f"{i}: {dev}")
            device = devices[0]
            print(f"Using device: {device}")

        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=4096, devicename=device)
        print("Playing sound... (click 'Stop Sound' to stop)")
        segment_size = 200
        sample_rate = 22050
        chunk_duration = 0.4
        chunk_samples = int(sample_rate * chunk_duration)

        # Define fade durations (in seconds)
        fade_in_duration = 0.01  # 10 ms fade-in
        fade_out_duration = 0.01  # 10 ms fade-out
        fade_in_samples = int(sample_rate * fade_in_duration)
        fade_out_samples = int(sample_rate * fade_out_duration)

        use_derivative = playback_mode != "No Derivative"
        values = global_freqs if use_derivative else global_raw_heights

        channel = pygame.mixer.Channel(0)
        sound_queue = queue.Queue()

        def audio_producer():
            sound_idx = 0
            phase = 0
            while sound_idx < len(values):
                if stop_audio_flag:
                    break
                end_idx = min(sound_idx + segment_size, len(values))
                start_val = values[sound_idx]
                end_val = values[end_idx - 1] if end_idx < len(values) else start_val
                t = np.linspace(0, chunk_duration, chunk_samples, endpoint=False)

                if playback_mode == "Smooth Transitions":
                    freqs = np.linspace(start_val, end_val, len(t))
                    phase_increment = 2 * np.pi * freqs / sample_rate
                    chunk_phase = phase + np.cumsum(phase_increment)
                else:
                    freqs = np.full(len(t), start_val)
                    phase_increment = 2 * np.pi * freqs / sample_rate
                    chunk_phase = phase + np.cumsum(phase_increment)

                phase = chunk_phase[-1]
                sound = 0.5 * np.sin(chunk_phase)

                # Apply fade-in and fade-out envelope
                envelope = np.ones(len(t))
                # Fade-in: first 10 ms
                if fade_in_samples > 0:
                    envelope[:fade_in_samples] = np.linspace(0, 1, fade_in_samples)
                # Fade-out: last 10 ms
                if fade_out_samples > 0:
                    envelope[-fade_out_samples:] = np.linspace(1, 0, fade_out_samples)
                sound = sound * envelope

                sound = (sound * 32767).astype(np.int16)
                sound = np.stack([sound, sound], axis=1)
                sound_queue.put(pygame.mixer.Sound(sound))
                sound_idx = end_idx

        def audio_consumer():
            while not stop_audio_flag:
                try:
                    sound = sound_queue.get(timeout=0.1)
                    channel.queue(sound)
                    while channel.get_busy() and not stop_audio_flag:
                        pygame.time.wait(10)
                except queue.Empty:
                    if not audio_producer_thread.is_alive():
                        break

        audio_producer_thread = threading.Thread(target=audio_producer)
        audio_producer_thread.start()
        audio_consumer()
        audio_producer_thread.join()
        pygame.mixer.quit()
        print("Sound stopped.")
    except Exception as e:
        print(f"Error during audio playback: {e}")
        pygame.mixer.quit()

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
        audio_thread.join()

def set_playback_mode(label):
    """Set the playback mode based on user selection."""
    global playback_mode
    playback_mode = label
    print(f"Playback mode set to: {playback_mode}")

def main():
    """Main function to load data, process it, and display the graph with interactive controls."""
    try:
        binary, file_path = load_binary_data()
        runs, heights = process_binary_data(binary)
        rate_of_change, smoothed_rate, display_smoothed_rate = compute_rate_of_change(heights)
        freqs = map_to_pentatonic(smoothed_rate, use_derivative=True)
        raw_freqs = map_to_pentatonic(heights, use_derivative=False)

        global global_heights, global_runs, global_freqs, global_raw_heights
        global_heights = heights
        global_runs = runs
        global_freqs = freqs
        global_raw_heights = raw_freqs

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

        ax_mode = plt.axes([0.81, 0.06, 0.1, 0.1])
        mode_selector = RadioButtons(ax_mode, ['Smooth Transitions', 'Sharp Tones', 'No Derivative'], active=0)
        mode_selector.on_clicked(set_playback_mode)

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
    except Exception as e:
        print(f"Error in main execution: {e}")
        exit(1)

if __name__ == "__main__":
    import sys
    main()