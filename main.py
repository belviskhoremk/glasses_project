import pvporcupine
import numpy as np
import sounddevice as sd
from detect_book import read_book , text_to_speech , play_audio
from environment_descriptor import describe_environment
token = "6hxcvUm5/sl99tTsPvp3GhEU25Jw1w3cnIL+mbbvyqXFDPp/KJvEdw=="
# Initialize Porcupine with your chosen wake word

KEYWORD_PATHS = [
    "/home/fitsum/Documents/projects/glasses_project/trigger words/read-book_en_linux_v3_0_0.ppn",
    "/home/fitsum/Documents/projects/glasses_project/trigger words/environment_en_linux_v3_0_0.ppn"
]

# Initialize Porcupine with your custom keywords
porcupine = pvporcupine.create(
    access_key=token,
    keyword_paths=KEYWORD_PATHS
)

def audio_callback(indata, frames, time, status):
    if status:
        print(status)

    pcm = np.frombuffer(indata, dtype=np.int16)
    keyword_index = porcupine.process(pcm)

    if keyword_index >= 0:
        if keyword_index == 0:
            print("Read book detected!")
            started = "start reading book, please wait a while to process"
            sound_data = text_to_speech(started)
            play_audio(sound_data)
            read_book()
            word = "open next page and instruct to read the new page "
            sound_data = text_to_speech(word)
            play_audio(sound_data)

        elif keyword_index == 1:
            print("Recognize environment detected!")
            sound_data = text_to_speech("detecting environment please wait a while to process the environment")
            play_audio(sound_data)
            describe_environment()
            sound_data = text_to_speech("Finished environment description")
            play_audio(sound_data)

            # Add your functionality for "recognize environment" here

try:
    with sd.InputStream(samplerate=porcupine.sample_rate,
                        blocksize=porcupine.frame_length,
                        dtype=np.int16,
                        channels=1,
                        callback=audio_callback):
        print("Listening for custom keywords...")
        print("Press Ctrl+C to stop.")
        while True:
            sd.sleep(100)  # Sleep to reduce CPU usage

except KeyboardInterrupt:
    print("Stopping...")

finally:
    porcupine.delete()


# from text_extract import extract_text
# from text_to_speech import text_to_speech, play_audio
#
#
# def extract_play(path):
#
#     try:
#
#         text = extract_text(path)
#         speech = text_to_speech(text)
#         play_audio(speech)
#         print("successfully played")
#
#     except Exception as e:
#         print(e)
#
# if __name__ == '__main__':
#     path = "sample-images/sample_book2.jpg"
#     extract_play(path)