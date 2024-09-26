import os
import struct
import pyaudio
import time
import numpy as np
import cv2
import mediapipe as mp
from pvrecorder import PvRecorder
import pvporcupine
import pvcobra
import whisper
from openai import OpenAI
import json
import soundfile as sf
import io
import threading
from yeelight import Bulb

# Yeelight bulbs setup
livingRoomBulb = Bulb("192.168.0.86")
bedRoomBulb = Bulb("192.168.0.85")

# MediaPipe pose initialization
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Global variables for camera and FPS
camera_active = False
COUNTER, FPS = 0, 0
START_TIME = time.time()
camera_thread = None  # Thread for running the camera

# OpenAI API setup
openai_api_key = 'YOURKEY'  # Replace with your OpenAI API key
client = OpenAI(api_key=openai_api_key)

# Variables for Wake-up word Recognition
access_key = 'YOURKEY'  # Replace with your Picovoice access key
keyword_paths = ['lemmy_jetson_1.ppn']  # Your wake word model path
library_path_porcupine = '/home/expc/.local/lib/python3.10/site-packages/pvporcupine/lib/jetson/cortex-a57-aarch64/libpv_porcupine.so'
audio_device_index = -1
sensitivities = [0.5] * len(keyword_paths)
speech_threshold = 1.3

# Function to generate audio output
def streamed_audio(input_text, model='tts-1', voice='nova'):
    try:
        print(f"Requesting TTS from OpenAI for text: {input_text}")
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=input_text,
            response_format="opus"
        )

        # Debugging: Check the size of the audio content from OpenAI
        print(f"Received audio response size: {len(response.content)} bytes")

        if len(response.content) == 0:
            print("Error: Empty audio response from OpenAI")
            return

        audio = pyaudio.PyAudio()

        def get_pyaudio_format(subtype):
            if subtype == 'PCM_16':
                return pyaudio.paInt16
            return pyaudio.paInt16

        buffer = io.BytesIO(response.content)
        buffer.seek(0)

        with sf.SoundFile(buffer, 'r') as sound_file:
            format = get_pyaudio_format(sound_file.subtype)
            channels = sound_file.channels
            rate = sound_file.samplerate

            # Increase chunk size to 4096 to reduce underruns
            chunk_size = 4096
            data = sound_file.read(chunk_size, dtype='int16')

            stream = audio.open(format=format, channels=channels, rate=rate, output=True)

            while len(data) > 0:
                try:
                    stream.write(data.tobytes())
                except IOError as e:
                    print(f"ALSA underrun occurred: {e}")
                    break  # Stop audio playback on underrun error
                data = sound_file.read(chunk_size, dtype='int16')

            stream.stop_stream()
            stream.close()

    except Exception as e:
        print(f"Error while streaming audio: {e}")

    finally:
        if audio is not None:
            audio.terminate()






# Camera control functions
def run_camera():
    global COUNTER, FPS, START_TIME, camera_active

    # Try to open the camera (camera index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        streamed_audio("Sorry, I could not activate the camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    print("Camera successfully turned on.")

    # Initialize MediaPipe Pose
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        START_TIME = time.time()
        while cap.isOpened() and camera_active:
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally for a selfie view
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            # Draw pose landmarklobals`. We recommend you start setting `weights_only=True` for any use case where you don't have fus on the image
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            COUNTER += 1
            if (time.time() - START_TIME) > 1:
                FPS = COUNTER / (time.time() - START_TIME)
                COUNTER = 0
                START_TIME = time.time()

            cv2.putText(image, f'FPS: {FPS:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('MediaPipe Pose', image)

            # Close window if 'Esc' is pressed
            if cv2.waitKey(5) & 0xFF == 27:  # Press Esc to exit
                camera_active = False
                break

        cap.release()
        cv2.destroyAllWindows()



def turn_on_camera():
    global camera_active, camera_thread
    if not camera_active:
        print("Turning on camera...")
        streamed_audio("Turning on camera")  # Audio feedback
        camera_active = True
        camera_thread = threading.Thread(target=run_camera)
        camera_thread.start()  # Start the camera in a separate thread

def turn_off_camera():
    global camera_active
    if camera_active:
        print("Turning off camera...")
        streamed_audio("Turning off camera")  # Audio feedback
        camera_active = False
        if camera_thread is not None:
            camera_thread.join()  # Wait for the camera thread to finish

def turn_on_light(location):
    try:
        if location == "living room":
            livingRoomBulb.turn_on()
        elif location == "bedroom":
            bedRoomBulb.turn_on()
    except Exception as e:
        print(f"Error turning on light in {location}: {e}")
        return f"Error turning on light in {location}: {e}"
    return f"Turned on the {location} light."

def turn_off_light(location):
    try:
        if location == "living room":
            livingRoomBulb.turn_off()
        elif location == "bedroom":
            bedRoomBulb.turn_off()
    except Exception as e:
        print(f"Error turning off light in {location}: {e}")
        return f"Error turning off light in {location}: {e}"
    return f"Turned off the {location} light."

def wakeUpWordRecognition():
    porcupine = None
    recorder = None
    try:
        porcupine = pvporcupine.create(
            access_key=access_key,
            keyword_paths=keyword_paths,
            sensitivities=sensitivities,
            library_path=library_path_porcupine  # Library path is crucial for porcupine to work
        )
        recorder = PvRecorder(device_index=audio_device_index, frame_length=porcupine.frame_length)
        recorder.start()
        print('LEMMY: Anytime call me ... (Waiting for Wake-up Word "Hey LEMMY!")')

        while True:
            pcm = recorder.read()
            result = porcupine.process(pcm)
            if result >= 0:
                print("LEMMY: Detected wake-up word!")
                return True
    finally:
        if recorder is not None:
            recorder.delete()
        if porcupine is not None:
            porcupine.delete()
    return False

def listen_for_voice():
    cobra = pvcobra.create(access_key=access_key)
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=cobra.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=cobra.frame_length)

    print("Listening...")
    start_time = time.time()
    frames = []

    while True:
        pcm = audio_stream.read(cobra.frame_length, exception_on_overflow=False)
        pcm = struct.unpack_from("h" * cobra.frame_length, pcm)
        voice_activity = cobra.process(pcm)

        if voice_activity > 0.3:
            frames.append(pcm)
            print("Voice started")
            while True:
                pcm = audio_stream.read(cobra.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h" * cobra.frame_length, pcm)
                voice_activity = cobra.process(pcm)
                frames.append(pcm)

                if voice_activity <= 0.3:
                    if len(frames) > 0:
                        last_voice_time = time.time()
                        while time.time() - last_voice_time <= speech_threshold:
                            pcm = audio_stream.read(cobra.frame_length, exception_on_overflow=False)
                            pcm = struct.unpack_from("h" * cobra.frame_length, pcm)
                            voice_activity = cobra.process(pcm)
                            if voice_activity > 0.3:
                                frames.append(pcm)
                                last_voice_time = time.time()
                            else:
                                frames.append(pcm)

                        print("Voice ended")
                        pcm_data = np.hstack(frames).astype(np.int16)
                        audio_stream.stop_stream()
                        audio_stream.close()
                        pa.terminate()
                        cobra.delete()
                        return pcm_data
                    else:
                        frames = []
                        break

        if time.time() - start_time > 5:
            print("No voice detected within the timeout period.")
            audio_stream.stop_stream()
            audio_stream.close()
            pa.terminate()
            cobra.delete()
            return None

def speech_to_text(pcm_data):
    # Convert audio to text using Whisper model
    whisper_model = whisper.load_model("base")
    audio_data = (pcm_data / 32768).astype(np.float32)
    result = whisper_model.transcribe(audio=audio_data, fp16=False)
    return result["text"]

def text_generate_GPT(messages):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        stream=False
    )
    response_message = response.choices[0].message
    messages.append(response_message)
    return messages, response_message.content

def main():
    personality = "You are a social robot for the elderly and your name is LEMMY."
    messages = [{"role": "system", "content": f"{personality}"}]
    camera_active = False

    while True:
        if wakeUpWordRecognition():
            pcm_data = listen_for_voice()
            if pcm_data is not None:
                user_input = speech_to_text(pcm_data)
                print(f"User said: {user_input.lower()}")

                # Handling flexible camera commands (on, off, camera)
                if "camera" in user_input.lower():
                    if "on" in user_input.lower() or "turn on" in user_input.lower():
                        if not camera_active:
                            streamed_audio("Turning on the camera.")
                            camera_active = True
                            camera_thread = threading.Thread(target=run_camera)
                            camera_thread.start()  # Start the camera in a separate thread
                        else:
                            streamed_audio("The camera is already on.")
                        continue  # Skip sending to GPT-4
                    
                    elif "off" in user_input.lower() or "turn off" in user_input.lower():
                        if camera_active:
                            streamed_audio("Turning off the camera.")
                            camera_active = False
                            if camera_thread is not None:
                                camera_thread.join()  # Safely stop the camera thread
                        else:
                            streamed_audio("The camera is already off.")
                        continue  # Skip sending to GPT-4

                # Handling flexible light commands (on, off, light)
                if "light" in user_input.lower():
                    if "on" in user_input.lower() or "turn on" in user_input.lower():
                        if "living room" in user_input.lower():
                            streamed_audio(turn_on_light("living room"))
                        elif "bedroom" in user_input.lower():
                            streamed_audio(turn_on_light("bedroom"))
                        continue  # Skip sending to GPT-4

                    elif "off" in user_input.lower() or "turn off" in user_input.lower():
                        if "living room" in user_input.lower():
                            streamed_audio(turn_off_light("living room"))
                        elif "bedroom" in user_input.lower():
                            streamed_audio(turn_off_light("bedroom"))
                        continue  # Skip sending to GPT-4

                # If the command is not related to hardware control, send it to GPT-4
                messages.append({"role": "user", "content": f"{user_input}"})
                messages, gpt_response = text_generate_GPT(messages)
                print(f"LEMMY: {gpt_response}")
                streamed_audio(gpt_response)  # Speak out GPT-4's response




if __name__ == '__main__':
    main()
