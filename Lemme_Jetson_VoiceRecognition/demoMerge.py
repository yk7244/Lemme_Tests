from openai import OpenAI
from pathlib import Path

import sounddevice as sd
import soundfile as sf
import pyaudio
import wave

# import os
# import sys
import urllib.request
import json
import requests
import pygame
from yeelight import Bulb

file_path = ".code/intentRecognitionDemo/"


# Mic Equipment Setting
RATE = 44100
CHUNK = int(RATE / 10)
BUFF = CHUNK * 10
FORMAT = pyaudio.paInt16
CHANNELS = 1  
DEVICE =24

# Clova Voice API Information
client_id = "Your Key"
client_secret = "Your Key"

# OpenAI API Information
client = OpenAI(api_key="Your Key")
personality = "너는 노인들을 위한 반려로봇이고 너 이름은 '래미'야. 너는 울산과학기술원에서 만들어졌어. 노인들이 알아들을 수 있게 짧고 친절하게 응답해."
messages = [{"role" : "system", "content" : f"{personality}"}]

### Kakao API Information
# 카카오 API 엑세스 토큰 kakao_code.json 불러오기
with open("kakao_code.json", "r") as fp:
    tokens = json.load(fp)    
# print(tokens)
print(tokens["access_token"])

# 카카오톡 나에게 메시지 보내기
def send_message():
    url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
    headers = {
        "Authorization": "Bearer " + tokens["access_token"]
    }
    template = {
        "object_type" : "list",
        "header_title" : "[LAMMY Test Message]",
        "header_link" : {
            "web_url" : "www.google.com",
            "mobile_web_url" : "www.google.com"
        },
        "contents" : [
            {
                "title" : "1. 낙상 감지 안내",
                "description" : "사용자의 낙상이 감지되었습니다.",
                "image_url" : "https://images.app.goo.gl/vYa4cAhrR3tfn19H6",
                "image_width" : 50, "image_height" : 50,
                "link" : {
                    "web_url" : "https://www.google.co.kr/search?q=national+park&source=lnms&tbm=nws",
                    "mobile_web_url" : "https://www.google.co.kr/search?q=national+park&source=lnms&tbm=nws"
                }
            },
            {
                "title" : "2. 119 호출 제안",
                "description" : "119로 연결해드릴까요?",
                "image_url" : "https://images.app.goo.gl/5Kq4FQMRkTdRVz756",
                "image_width" : 50, "image_height" : 50,
                "link" : {
                    "web_url" : "https://www.google.co.kr/search?q=deep+learning&source=lnms&tbm=nws",
                    "mobile_web_url" : "https://www.google.co.kr/search?q=deep+learning&source=lnms&tbm=nws"
                }
            }
        ],
        "buttons" : [
            {
                "title" : "119에 연결",
                "link" : {
                    "web_url" : "www.google.com",
                    "mobile_web_url" : "www.google.com"
                }
            }
        ]
    }
    data = {
        "template_object" : json.dumps(template)
    }
    response = requests.post(url, data=data, headers=headers)
    # print(response.status_code)
    if response.json().get('result_code') == 0:
        print('메시지를 성공적으로 보냈습니다.')
    else:
        print('메시지를 성공적으로 보내지 못했습니다. 오류메시지 : ' + str(response.json()))

# # 친구 목록 가져오기
# url = "https://kapi.kakao.com/v1/api/talk/friends" #친구 목록 가져오기
# header = {"Authorization": 'Bearer ' + tokens["access_token"]}
# result = json.loads(requests.get(url, headers=header).text)
# friends_list = result.get("elements")

# # 친구 목록 중 0번째 리스트의 친구 'uuid'
# friend_id = friends_list[0].get("uuid")

# 친구에게 카카오톡 메시지 보내기
# def send_message():
#     url= "https://kapi.kakao.com/v1/api/talk/friends/message/default/send"
#     header = {"Authorization": 'Bearer ' + tokens["access_token"]}
#     data={
#         'receiver_uuids': '["{}"]'.format(friend_id),
#         "template_object": json.dumps({
#             "object_type":"text",
#             "text":"[LEMMY Test] 응급상황 발생!",
#             "link":{
#                 "web_url" : "https://expc.unist.ac.kr",
#                 "mobile_web_url" : "https://expc.unist.ac.kr"
#             },
#             "button_title": "119 호출하기"
#         })
#     }
#     response = requests.post(url, headers=header, data=data)
#     response.status_code

### IoT Bulb Information
# Bulbs List (EXPC_Lab WIFI)
livingRoomBulb = Bulb("192.168.0.86")
bedRoomBulb = Bulb("192.168.0.85")

# Function Turn on the IoT light
def turn_on_light(location):
    if location == "거실":
        livingRoomBulb.turn_on()
    elif location == "침실":
        bedRoomBulb.turn_on()

    # placeBulb.turn_on()

# Funtion Turn off the IoT light
def turn_off_light(location):
    if location == "거실":
        livingRoomBulb.turn_off()
    elif location == "침실":
        bedRoomBulb.turn_off()
        
    # placeBulb.turn_off()

# ChatGPT 호출 Tool 목록
tools = [
  {
    "type": "function",
    "function": {
      "name": "turn_on_light",
      "description": "방 이름을 입력해서 그곳의 불 켜기",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "방 이름, 예를 들면 침실, 거실",
          },
        },
        "required": ["location"],
      },
    },
  },
  {
    "type": "function",
    "function": {
      "name": "turn_off_light",
      "description": "방 이름을 입력해서 그곳의 불 끄기",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "방 이름, 예를 들면 침실, 거실",
          },
        },
        "required": ["location"],
      },
    },
  },
  {
    "type": "function",
    "function": {
      "name": "send_message",
      "description": "응급상황이라고 판단되면 메시지 보내기",
      # "parameters": {
      #   "type": "object",
      #   "properties": {
      #     "emergency": {
      #       "type": "bool",
      #       "description": "응급 상황인지 아닌지 판정",
      #     },
      #   },
      #   "required": ["emergency"],
      # },
    },
  }
]

# 한국어 음성 인식
def speech_recognition():
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=DEVICE,
                frames_per_buffer=CHUNK)

    print('start recording')

    frames = []
    seconds = 5 # 녹음 시간
    for i in range(0,int(RATE / CHUNK * seconds)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    print('record stopped')

    stream.stop_stream()
    stream.close()
    p.terminate()
    #wf = wave.open(file_path + "realtime_input.wav",'wb')
    wf = wave.open("realtime_input.wav",'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # wav_file = file_path + "realtime_input.wav"

    #audio_file = open(file_path + "realtime_input.wav", "rb")
    audio_file = open("realtime_input.wav", "rb")



    # whisper 모델에 음원파일 전달하기
    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    print("User: " + transcript.text)
    return transcript.text

# # 한국어 자연어 처리
# def generate_text():
#     response = client.chat.completions.create(
#     model="ft:gpt-3.5-turbo-0613:personal::8K8T6bot",       # Finetuning model
#     messages=messages
#     )
#     print(response.choices[0].message)
#     print("LEMMY: "+ response.choices[0].message.content)

#     bot_response = response.choices[0].message.content
#     messages.append({"role" : "assistant", "content" : f"{bot_response}"})
#     return bot_response

# 한국어 자연어 처리 및 명령어 인식
def generate_text(user_input):
    messages.append({"role" : "user", "content" : user_input})
    # print(messages)
    response = client.chat.completions.create(
    # model="ft:gpt-3.5-turbo-0613:personal::8K8T6bot",       # Finetuning model
    model = 'gpt-4-turbo-preview',
    n = 1,
    messages=messages,
    tools=tools,
    tool_choice="auto"
    )
    # print(response)


    completion_reponse = response.choices[0].message
    bot_response = completion_reponse.content
    
    # print(bot_response)



    if completion_reponse.tool_calls: # 응답이 함수 호출인지 확인하기
        # 호출할 함수 이름을 지정 
        available_functions = {"turn_on_light": turn_on_light, "turn_off_light": turn_off_light, "send_message": send_message}

        # 함수 이름 추출
        # print(completion_reponse)
        for tool_call in completion_reponse.tool_calls:
            # function_name = completion_reponse.tool_calls[0].function.name
            function_name = tool_call.function.name

            
            # 호출할 함수 선택
            function_to_call = available_functions[function_name]

            # 함수 호출 및 반환 결과 받기
            if function_name == 'send_message':
              function_to_call()
              bot_response = "응급 메시지를 전송합니다!"
            else:
              function_to_call(
                  location=json.loads(tool_call.function.arguments).get('location')
              )
              bot_response = '뿅!'
    
    # # tool calling 시 응답 설정
    # if not isinstance(bot_response, str):
    #     bot_response = '뿅'
    
    messages.append({"role" : "assistant", "content" : f"{bot_response}"})


    print("LEMMY: "+ bot_response)

    return bot_response

# # 한국어 명령어 인식 및 자유 대화 진행
# def run_conversation(user_query):
#     # 사용자 입력
#     messages.append({"role" : "user", "content" : user_query})
#     print(messages)
#     completion = client.chat.completions.create(
#     # model="ft:gpt-3.5-turbo-0613:personal::8K8T6bot",
#     model='gpt-4-1106-preview',
#     messages=messages,
#     tools=tools,
#     tool_choice="auto"
#     )
#     completion_reponse = completion.choices[0].message
#     print(completion_reponse)
#     # print(completion_reponse)

#     bot_response = completion.choices[0].message.content
#     messages.append({"role" : "assistant", "content" : f"{bot_response}"})
#     print("LEMMY: " + bot_response)


#     if completion_reponse.tool_calls: # 응답이 함수 호출인지 확인하기
#         # 호출할 함수 이름을 지정 
#         available_functions = {"turn_on_light": turn_on_light, "turn_off_light": turn_off_light}

#         # 함수 이름 추출
#         # print(completion_reponse)
#         for tool_call in completion_reponse.tool_calls:
#             # function_name = completion_reponse.tool_calls[0].function.name
#             function_name = tool_call.function.name

            
#             # 호출할 함수 선택
#             fuction_to_call = available_functions[function_name]

#             # 함수 호출 및 반환 결과 받기
#             fuction_to_call(
#                 location=json.loads(tool_call.function.arguments).get('location')
#             )
#         # print(completion_reponse.tool_calls[0].function.name)
            
#     return completion_reponse.content

# 한국어 음성 합성
def generate_audio(text):
    encText = urllib.parse.quote(text)

    ### Set Parameter
    speaker = "vdain"
    volume = "VOLUME"       # -5 ~ +5 / -5:0.5배 낮은 볼륨, +5: 1.5배 큰 볼륨
    speed = "SPEED"     # -5 ~ +5 / -5: 2배 빠른 속도, +5: 0.5배 느린 속도
    pitch = "PITCH"     # -5 ~ +5 / -5: 1.2배 높은 피치, +5: 0.8배 낮은 피치
    emotion = "0"     # 0:중립, 1: 슬픔, 2: 기쁨, 3: 분노
    emotion_strength = "EMOTION_STRENGTH"       # 0: 약함, 1: 보통, 2: 강함

    #speech_file_path = Path(__file__).parent / "clovaTemp.mp3"
    speech_file_path = "clovaTemp.mp3"



    data = "speaker=" + speaker + "&volume=0&speed=0&pitch=0&format=mp3&emotion=" + emotion + "&emotion_strength=2&text=" + encText
    url = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"
    request = urllib.request.Request(url)
    request.add_header("X-NCP-APIGW-API-KEY-ID",client_id)
    request.add_header("X-NCP-APIGW-API-KEY",client_secret)
    response = urllib.request.urlopen(request, data=data.encode('utf-8'))
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        with open('clovaTemp.mp3', 'wb') as f:
            f.write(response_body)
        # Play the MP3 file using pygame
        play_mp3('clovaTemp.mp3')
    else:
        print("Error Code(Clova Voice):" + rescode)


def play_mp3(file_path):
    # Initialize pygame mixer
    pygame.mixer.init()
    # Load the MP3 file
    pygame.mixer.music.load(file_path)
    # Play the MP3 file
    pygame.mixer.music.play()
    # Wait for playback to finish
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)



def main():
    while True:
        print("======================")
        user_input = speech_recognition()
        # messages.append({"role" : "user", "content" : f"{user_input}"})
        bot_response = generate_text(user_input)
        # bot_response = run_conversation(user_input)
        #generate_audio(bot_response)

if __name__ == "__main__":
    main()
