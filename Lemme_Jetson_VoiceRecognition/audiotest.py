import pyaudio

# Check the input devices and their supported channels
def list_input_devices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')

    for i in range(0, numdevices):
        if p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'),
                  " - Channels: ", p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels'))

    p.terminate()

list_input_devices()
