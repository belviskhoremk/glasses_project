"""
In this file, I developed tqo functions. One for converting the text into audio
And one for playing the audio

"""
import pytesseract
from gtts import gTTS
import pygame
import os


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
    THe function plays the audio file using pygame library
    :param audio_file: The audio file to play
    :return: None
    """
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue
    pygame.mixer.quit()

    # Clean up the audio file
    if os.path.exists(audio_file):
        os.remove(audio_file)



if __name__ == '__main__':
    audio_file = text_to_speech("My name is Elvis. And I am a developer!")
    # Play the audio
    play_audio(audio_file)


