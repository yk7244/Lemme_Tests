import requests
import json
import time
import numpy as np
import pyaudio
import struct
import io
from openai import OpenAI
from pvrecorder import PvRecorder
import pvporcupine
import pvcobra
import whisper
import soundfile as sf

# Initialize OpenAI client
client = OpenAI(api_key="")

# Variables for Wake-up word Recognition
access_key = ''
keyword_paths = ['lemmy_jetson_1.ppn']
audio_device_index = -1
sensitivities = [0.5] * len(keyword_paths)
library_path_porcupine = '/home/expc/.local/lib/python3.10/site-packages/pvporcupine/lib/jetson/cortex-a57-aarch64/libpv_porcupine.so'


# Variables for recording (voice activity detection)
speech_threshold = 1.3

# Load Whisper model
model = whisper.load_model("small")

def get_weather_info(city, apikey, lang="en", units="metric"):
    api = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={apikey}&lang={lang}&units={units}"
    result = requests.get(api)
    if result.status_code != 200:
        return None
    data = json.loads(result.text)
    weather_info = {
        "location": data["name"],
        "weather": data["weather"][0]["description"],
        "temperature": {
            "current": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "min": data["main"]["temp_min"],
            "max": data["main"]["temp_max"]
        }
    }
    return weather_info

def get_current_weather(location, unit="celsius"):
    apikey = "1"
    lang = "en"
    weather_info = get_weather_info(location, apikey, lang)
    
    if weather_info is None:
        return json.dumps({"error": f"Unable to fetch weather data for {location}"})
    
    return json.dumps({
        "location": weather_info["location"],
        "temperature": str(weather_info["temperature"]["current"]),
        "weather": weather_info["weather"],
        "unit": unit
    })

def turn_on_air_conditioner():
    print("******TURN ON THE AIR CONDITIONER!!!******")
    return json.dumps({"status": "Air conditioner turned on"})

def end_conversation():
    return json.dumps({"status": "Ending conversation"})

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
        print("=======(Waiting for Wake-up Word (Hey Lemmy!)) (press Ctrl+C to exit)=======")
        # print('LEMMY: Anytime call me ...')
        
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

def streamed_audio(input_text, model='tts-1', voice='nova'):
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=input_text,
        response_format="wav"
    )

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
        
        stream = audio.open(format=format, channels=channels, rate=rate, output=True)
        chunk_size = 1024
        data = sound_file.read(chunk_size, dtype='int16')
        
        while len(data) > 0:
            stream.write(data.tobytes())
            data = sound_file.read(chunk_size, dtype='int16')
        
        stream.stop_stream()
        stream.close()

    audio.terminate()

def generate_response(messages):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "turn_on_air_conditioner",
                "description": "Turn on the air conditioner",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "end_conversation",
                "description": "End the current conversation session with a friendly message such as goodbye, see you, take care, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                },
            },
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.8
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        function_messages = []
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            if function_name == "get_current_weather":
                function_response = get_current_weather(
                    location=function_args.get("location"),
                    unit=function_args.get("unit", "celsius")
                )
            elif function_name == "turn_on_air_conditioner":
                function_response = turn_on_air_conditioner()
            elif function_name == "end_conversation":
                function_response = end_conversation()
            else:
                raise ValueError(f"Unknown function: {function_name}")

            function_messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            })

        messages.append(response_message)
        messages.extend(function_messages)

        # Generate a new response based on function results
        final_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.8
        )
        
        final_message = final_response.choices[0].message.content
        
        if function_name == "end_conversation":
            return "END_CONVERSATION", messages, final_message
        else:
            return final_message, messages

    else:
        return response_message.content, messages

def main():
    personality = "You are a companion robot for the elderly and your name is LEMMY. You were created at UNIST. You respond in short, friendly sentences that the elderly can understand. You can provide weather information for any city in the world. When you sense that the conversation is coming to a natural end, use the end_conversation function to conclude the chat politely."
    messages = [{"role": "system", "content": personality}]

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
                    print(f'USER: {user_input}')
                    messages.append({"role": "user", "content": user_input})
                    bot_response, messages, *extra = generate_response(messages)
                    
                    if bot_response == "END_CONVERSATION":
                        final_message = extra[0]  # Get the generated goodbye message
                        print(f'LEMMY: {final_message}')
                        streamed_audio(final_message)
                        dialogueEnd = True
                        wakeUpDetect = False
                        break
                    else:
                        print(f'LEMMY: {bot_response}')
                        streamed_audio(bot_response)
                        messages.append({"role": "assistant", "content": bot_response})
                    print("="*20)
                else:
                    print("No voice detected. Going back to wake-up word detection.")
                    wakeUpDetect = False
                    break

if __name__ == "__main__":
    main()