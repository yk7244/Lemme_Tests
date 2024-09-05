import os
import struct
import pyaudio
import time
import numpy as np
from pvrecorder import PvRecorder
import pvporcupine
import pvcobra
import whisper
from openai import OpenAI
import json
import soundfile as sf
import io
from yeelight import Bulb  # Importing Yeelight functionality

# YOUR_API_KEY for wake-up word recognition and NLP
access_key = 'your key'



# Yeelight bulbs setup
livingRoomBulb = Bulb("192.168.0.86")
bedRoomBulb = Bulb("192.168.0.85")

# Functions for turning lights on and off
def turn_on_light(location):
    if location == "거실":
        livingRoomBulb.turn_on()
    elif location == "침실":
        bedRoomBulb.turn_on()

def turn_off_light(location):
    if location == "거실":
        livingRoomBulb.turn_off()
    elif location == "침실":
        bedRoomBulb.turn_off()

# Variables for Wake-up word Recognition
keyword_paths = ['lemmy_jetson_1.ppn']
audio_device_index = -1
sensitivities = [0.5] * len(keyword_paths)
library_path_porcupine = '/home/expc/.local/lib/python3.10/site-packages/pvporcupine/lib/jetson/cortex-a57-aarch64/libpv_porcupine.so'


# Variables for recording (voice activity detection)
speech_threshold = 1.3

# Variables for Loading Whisper model
model = whisper.load_model("base")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

def wakeUpWordRecognition():
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

        print('LEMMY: Anytime call me ... (Waiting for Wake-up Word "Hey LEMMY!") (press Ctrl+C to exit)')
        
        while True:
            pcm = recorder.read()
            result = porcupine.process(pcm)

            if result >= 0:
                print("LEMMY: Detected wake-up word!")
                return True
    except pvporcupine.PorcupineError as e:
        print("Failed to initialize Porcupine: ", e)
        return False
    except KeyboardInterrupt:
        print('Stopping ...')
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
    timeout = 5
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

        if time.time() - start_time > timeout:
            print("No voice detected within the timeout period.")
            audio_stream.stop_stream()
            audio_stream.close()
            pa.terminate()
            cobra.delete()
            return None

def speech_to_text(pcm_data):
    audio_data = (pcm_data / 32768).astype(np.float32)
    result = model.transcribe(audio=audio_data, fp16=False)
    return result["text"]

def text_generate_GPT(messages, tools):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    response_message = response.choices[0].message
    messages.append(response_message)
    return messages, response_message.tool_calls, response_message.content

def end_conversation():
    return json.dumps({"message": "Thank you for chatting with me! Have a wonderful day!"})

def streamed_audio(input_text, model='tts-1', voice='nova'):
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=input_text,
        response_format="opus"
    )

    audio = pyaudio.PyAudio()

    def get_pyaudio_format(subtype):
        if subtype == 'PCM_16':
            return pyaudio.paInt16
        return pyaudio.paInt16

    try:
        buffer = io.BytesIO(response.content)
        buffer.seek(0)

        with sf.SoundFile(buffer, 'r') as sound_file:
            format = get_pyaudio_format(sound_file.subtype)
            channels = sound_file.channels
            rate = sound_file.samplerate
            
            stream = audio.open(format=format, channels=channels, rate=rate, output=True)
            chunk_size = 1024
            data = sound_file.read(chunk_size, dtype='int16')
            
            while len(data) > 0:
                stream.write(data.tobytes())
                data = sound_file.read(chunk_size, dtype='int16')
            
            stream.stop_stream()
            stream.close()

    except Exception as e:
        print(f"Error while streaming audio: {e}")
    finally:
        audio.terminate()

def main():
    personality = "You are a social robot for the elderly and your name is LEMMY."
    messages = [{"role": "system", "content": f"{personality}"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "end_conversation",
                "description": "End the current conversation session with a friendly message.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "turn_on_light",
                "description": "Turn on the light in a specified location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "Location (e.g., 거실, 침실)"}
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "turn_off_light",
                "description": "Turn off the light in a specified location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "Location (e.g., 거실, 침실)"}
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    wakeUpDetect = False
    dialogueEnd = False

    while True:
        if not wakeUpDetect:
            wakeUpDetect = wakeUpWordRecognition()
        else:
            while not dialogueEnd:
                pcm_data = listen_for_voice()
                if isinstance(pcm_data, np.ndarray):
                    user_input = speech_to_text(pcm_data)
                    print(f'User: {user_input}')
                    messages.append({"role": "user", "content": f"{user_input}"})
                    messages, tool_calls, lemmy_response = text_generate_GPT(messages, tools)
                    
                    if lemmy_response:
                        print(f'LEMMY: {lemmy_response}')
                        streamed_audio(lemmy_response)
                    
                    if tool_calls:
                        available_functions = {
                            "end_conversation": end_conversation,
                            "turn_on_light": turn_on_light,
                            "turn_off_light": turn_off_light
                        }
                        for tool_call in tool_calls:
                            function_name = tool_call.function.name
                            function_to_call = available_functions[function_name]
                            function_response = function_to_call(**json.loads(tool_call.function.arguments))
                            messages.append({"role": "assistant", "content": function_response})
                        dialogueEnd = True
                        wakeUpDetect = False
                        break
                else:
                    print("No voice detected. Going back to wake-up word detection.")
                    wakeUpDetect = False
                    break

if __name__ == '__main__':
    main()
