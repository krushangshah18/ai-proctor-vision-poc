import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import time

DURATION = 3  # seconds to record
SAMPLE_RATE = 16000


def list_audio_devices():
    print("\nAvailable Audio Input Devices:\n")
    devices = sd.query_devices()

    input_devices = []
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append((i, device))
            print(f"Index: {i}")
            print(f"Name: {device['name']}")
            print(f"Input Channels: {device['max_input_channels']}")
            print(f"Default Sample Rate: {device['default_samplerate']}")
            print("-" * 40)

    return input_devices


def test_device(device_index):
    print(f"\nTesting device {device_index}... Recording {DURATION}s")

    recording = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        device=device_index,
        dtype='int16'
    )

    sd.wait()

    filename = f"test_device_{device_index}.wav"
    wav.write(filename, SAMPLE_RATE, recording)

    print(f"Saved recording to {filename}")


def main():
    devices = list_audio_devices()

    for index, device in devices:
        try:
            test_device(index)
        except Exception as e:
            print(f"Device {index} failed: {e}")


if __name__ == "__main__":
    main()