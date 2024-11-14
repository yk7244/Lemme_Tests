import time
import numpy as np
import pvporcupine
from pvrecorder import PvRecorder
import whisper
import torch
import serial
import pvcobra
import pyaudio
import struct

# Initialize necessary variables
access_key = ''  # Add your Porcupine access key here
keyword_paths = ['lemmy_jetson_1.ppn']  # Path to the wake word model file
audio_device_index = -1  # Set audio device index (usually -1 for default)
sensitivities = [0.5] * len(keyword_paths)  # Sensitivity for wake word detection
library_path_porcupine = ''  # Path to Porcupine library
device = "cuda"  # Use GPU if available
model = whisper.load_model("small").to(device)  # Load Whisper model

def wake_up_word_detection():
    """
    Function to detect the wake-up word using Porcupine.
    """
    porcupine = None
    recorder = None
    try:
        porcupine = pvporcupine.create(
            access_key=access_key,
            keyword_paths=keyword_paths,
            sensitivities=sensitivities,
            library_path=library_path_porcupine,
        )
        
        recorder = PvRecorder(device_index=audio_device_index, frame_length=porcupine.frame_length)
        recorder.start()
        print("Listening for wake-up word...")
        
        while True:
            pcm = recorder.read()
            result = porcupine.process(pcm)

            if result >= 0:
                print("Detected wake-up word!")
                return True
    except pvporcupine.PorcupineError as e:
        print("Failed to initialize Porcupine:", e)
        return False
    finally:
        if recorder is not None:
            recorder.delete()
        if porcupine is not None:
            porcupine.delete()
    return False

def listen_for_voice():
    """
    Listens for voice input and returns the PCM data if voice activity is detected.
    Stops recording when there is silence for 3 seconds.
    """
    cobra = pvcobra.create(access_key=access_key)
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=cobra.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=cobra.frame_length)

    print("Listening for voice...")
    frames = []
    start_time = time.time()
    silence_start = None
    voice_detected = False

    while True:
        pcm = audio_stream.read(cobra.frame_length, exception_on_overflow=False)
        pcm = struct.unpack_from("h" * cobra.frame_length, pcm)
        voice_activity = cobra.process(pcm)

        # Append frames if voice detected
        if voice_activity > 0.3:
            frames.append(pcm)
            voice_detected = True
            silence_start = time.time()  # Reset silence timer
        elif voice_detected and (silence_start is None):
            silence_start = time.time()  # Start silence timer if no voice

        # Stop after 3 seconds of silence
        if silence_start and (time.time() - silence_start > 3):
            print("Silence detected for 3 seconds. Stopping recording.")
            break

        # Timeout for no voice detected in 5 seconds after wake-up
        if not voice_detected and (time.time() - start_time > 5):
            print("No voice detected within 5 seconds after wake-up.")
            audio_stream.stop_stream()
            audio_stream.close()
            pa.terminate()
            cobra.delete()
            return None

    pcm_data = np.hstack(frames).astype(np.int16) if frames else None
    audio_stream.stop_stream()
    audio_stream.close()
    pa.terminate()
    cobra.delete()
    return pcm_data

def speech_to_text(pcm_data):
    """
    Function to transcribe PCM data using the Whisper model.
    """
    audio_data = (pcm_data / 32768).astype(np.float32)
    result = model.transcribe(audio=audio_data, fp16=True if device == "cuda" else False)
    return result["text"]

def send_serial_data(data, port='/dev/ttyUSB0', baudrate=9600):
    """
    Function to send data over serial to an external device.
    
    Parameters:
    - data: The data to be sent as a string.
    - port: The serial port to use (default '/dev/ttyUSB0').
    - baudrate: The communication baud rate (default 9600).
    """
    try:
        with serial.Serial(port, baudrate, timeout=1) as ser:
            ser.write(data.encode())  # Send the data as bytes
            print(f"Sent data: {data}")
    except serial.SerialException as e:
        print("Error in serial communication:", e)

# Main loop to repeatedly detect wake word, listen, transcribe, and send data
if __name__ == "__main__":
    while True:
        # Step 1: Wait for the wake-up word
        if wake_up_word_detection():
            print("Wake-up word detected. Listening for speech...")

            # Step 2: Capture audio and convert it to text
            pcm_data = listen_for_voice()
            if pcm_data is not None:
                text = speech_to_text(pcm_data)
                print("Transcribed Text:", text)
                
                # Step 3: Send the transcribed text over serial to the Orange Pi
                send_serial_data(text, port='/dev/ttyUSB0', baudrate=9600)
                print("Data sent to Android board.")
            else:
                print("Returning to wake-up word detection mode due to timeout.")
