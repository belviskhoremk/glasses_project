"""
In this file, I developed tqo functions. One for converting the text into audio
And one for playong the audio

"""
import pytesseract
from gtts import gTTS
import pygame
import os

# Tesseract path (assume installed in /usr/bin/tesseract on Ubuntu)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def text_to_speech(text, lang='en'):
    """
    The function is used to convert a text into an audio file
    :param text: The text to be converted into audio(.mp3) format
    :param lang: The language in which the text is written and audio played
    :return: The audio file in mp3 format
    """
    tts = gTTS(text=text, lang=lang, tld='com')
    audio_file = 'output.mp3'
    tts.save(audio_file)
    return audio_file


def play_audio(audio_file):
    """
    THe function plays the audio file
    :param audio_file: The audio file to play
    :return: None
    """
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue
    pygame.mixer.quit()



# audio_file = text_to_speech("My name is Elvis. And I am a developer!")
#
# # Play the audio
# play_audio(audio_file)

# Clean up the audio file
# if os.path.exists(audio_file):
#     os.remove(audio_file)
