import os
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

# Suppress ALSA errors
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["ALSA_CARD"] = "default"

# Initialize necessary variables
access_key = ''  # Add your Porcupine access key here
keyword_paths = ['lemmy_jetson_1.ppn']  # Path to the wake word model file
audio_device_index = -1  # Set audio device index (usually -1 for default)
sensitivities = [0.5] * len(keyword_paths)  # Sensitivity for wake word detection
library_path_porcupine = '/home/expc/.local/lib/python3.10/site-packages/pvporcupine/lib/jetson/cortex-a57-aarch64/libpv_porcupine.so'  # Path to Porcupine library
device = "cuda"  # Use GPU if available
model = whisper.load_model("small").to(device)  # Load Whisper model

# Serial communication setup
uart_port = '/dev/ttyTHS1'  # UART port for Jetson
baud_rate = 9600  # Communication baudrate

# Function to send data via the specified protocol
def send_protocol_data(port, baudrate, header1, header2, command, data, chunk_size=64):
    try:
        with serial.Serial(port, baudrate, timeout=1) as ser:
            data_bytes = data.encode() if isinstance(data, str) else data
            data_length = len(data_bytes)
            message = f"{header1}{header2}{command}{data_length:02X}".encode() + data_bytes + b'>'

            # Send data in chunks
            for i in range(0, len(message), chunk_size):
                ser.write(message[i:i + chunk_size])
                time.sleep(0.1)  # Allow UART buffer to clear
                print(f"Sent chunk: {message[i:i + chunk_size]}")
    except serial.SerialException as e:
        print(f"Error: {e}")

# Function to receive data via the specified protocol
def receive_protocol_data(port, baudrate, timeout=2):
    try:
        with serial.Serial(port, baudrate, timeout=1) as ser:
            print("Waiting for data from AP...")
            start_time = time.time()
            response = b""

            while time.time() - start_time < timeout:
                if ser.in_waiting:
                    response += ser.read(ser.in_waiting)
                    time.sleep(0.1)

                # Check for valid response format
                if response.startswith(b'<') and response.endswith(b'>'):
                    raw_message = response.decode()
                    print(f"Received: {raw_message}")
                    header1 = raw_message[0]
                    header2 = raw_message[1:3]
                    command = raw_message[3:6]
                    data_length = int(raw_message[6:8], 16)
                    data = raw_message[8:-1]

                    return {
                        "header1": header1,
                        "header2": header2,
                        "command": command,
                        "data_length": data_length,
                        "data": data,
                    }

            print("Timeout or incomplete data.")
            return None
    except serial.SerialException as e:
        print(f"Error: {e}")
        return None

# Wake-up word detection function
def wake_up_word_detection():
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
        send_protocol_data(uart_port, baud_rate, '<', 'JA', 'MOD', '10')  # Enter wake-up detection mode
        while True:
            pcm = recorder.read()
            result = porcupine.process(pcm)
            if result >= 0:
                print("Detected wake-up word!")
                send_protocol_data(uart_port, baud_rate, '<', 'JA', 'MOD', '11')  # Wake word detected
                return True
    except pvporcupine.PorcupineError as e:
        print("Failed to initialize Porcupine:", e)
        send_protocol_data(uart_port, baud_rate, '<', 'JA', 'MOD', '12')  # Detection failed
        return False
    finally:
        if recorder is not None:
            recorder.delete()
        if porcupine is not None:
            porcupine.delete()
    return False

# Function to listen for voice input
def listen_for_voice():
    cobra = pvcobra.create(access_key=access_key)
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=cobra.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=cobra.frame_length)

    print("Listening for voice...")
    send_protocol_data(uart_port, baud_rate, '<', 'JA', 'MOD', '12')  # Enter listening mode
    frames = []
    start_time = time.time()
    silence_start = None
    voice_detected = False

    while True:
        pcm = audio_stream.read(cobra.frame_length, exception_on_overflow=False)
        pcm = struct.unpack_from("h" * cobra.frame_length, pcm)
        voice_activity = cobra.process(pcm)

        if voice_activity > 0.3:
            frames.append(pcm)
            voice_detected = True
            silence_start = time.time()  # Reset silence timer
        elif voice_detected and (silence_start is None):
            silence_start = time.time()  # Start silence timer if no voice

        if silence_start and (time.time() - silence_start > 3):
            print("Silence detected for 3 seconds. Stopping recording.")
            break

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

# Speech-to-text function
def speech_to_text(pcm_data):
    audio_data = (pcm_data / 32768).astype(np.float32)
    result = model.transcribe(audio=audio_data, fp16=True if device == "cuda" else False)
    return result["text"]

# Main loop for real AP board communication
if __name__ == "__main__":
    while True:
        # Receive a command from the AP
        protocol_data = receive_protocol_data(uart_port, baud_rate)
        if protocol_data:
            if protocol_data["header2"] == "AJ" and protocol_data["command"] == "STA":
                print("Received status request. Responding with Jetson status...")
                # Send a response back to the AP
                send_protocol_data(uart_port, baud_rate, '<', 'JA', 'STA', '0')

        # Simulate wake-up word detection
        if wake_up_word_detection():
            print("Wake-up word detected. Listening for speech...")
            # Capture real PCM data and process it with STT
            pcm_data = listen_for_voice()
            if pcm_data is not None:
                text = speech_to_text(pcm_data)
                print("Transcribed Text:", text)
                send_protocol_data(uart_port, baud_rate, '<', 'JA', 'SND', text)
            else:
                print("No speech detected or timeout occurred.")

        time.sleep(2)  # Wait before the next iteration