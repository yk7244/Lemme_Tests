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
import threading

# MediaPipe pose initialization
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Global variables for camera and FPS
camera_active = False
COUNTER, FPS = 0, 0
START_TIME = time.time()
camera_thread = None  # Thread for running the camera

# Variables for Wake-up word Recognition
access_key = 'your key'  # Replace with your Picovoice access key
keyword_paths = ['lemmy_jetson_1.ppn']  # Your wake word model path
library_path_porcupine = '/home/expc/.local/lib/python3.10/site-packages/pvporcupine/lib/jetson/cortex-a57-aarch64/libpv_porcupine.so'
audio_device_index = -1
sensitivities = [0.5] * len(keyword_paths)
speech_threshold = 1.3

# Camera control function
def run_camera():
    global COUNTER, FPS, START_TIME, camera_active
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

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

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

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

            if cv2.waitKey(5) & 0xFF == 27:  # Exit if 'Esc' is pressed
                break

        cap.release()
        cv2.destroyAllWindows()

def turn_on_camera():
    global camera_active, camera_thread
    if not camera_active:
        print("Turning on camera...")
        camera_active = True
        camera_thread = threading.Thread(target=run_camera)
        camera_thread.start()  # Start the camera in a separate thread

def turn_off_camera():
    global camera_active
    if camera_active:
        print("Turning off camera...")
        camera_active = False
        if camera_thread is not None:
            camera_thread.join()  # Wait for the camera thread to finish

def wakeUpWordRecognition():
    porcupine = None
    recorder = None
    try:
        porcupine = pvporcupine.create(
            access_key=access_key,
            keyword_paths=keyword_paths,
            sensitivities=sensitivities,
            library_path=library_path_porcupine  
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
    # Simulated speech-to-text functionality:
    # You can replace this with actual speech-to-text processing if needed.
    audio_data = (pcm_data / 32768).astype(np.float32)
    if np.random.random() < 0.5:  # Simulated random recognition for demo purposes
        return "turn on camera"
    else:
        return "turn off camera"

def main():
    while True:
        if wakeUpWordRecognition():
            pcm_data = listen_for_voice()
            if pcm_data is not None:
                user_input = speech_to_text(pcm_data)
                print(f"User said: {user_input}")

                # Camera control based on voice commands
                if "turn on camera" in user_input.lower():
                    turn_on_camera()
                elif "turn off camera" in user_input.lower():
                    turn_off_camera()

if __name__ == '__main__':
    main()
