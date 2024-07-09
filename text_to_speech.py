"""
In this file, I developed tqo functions. One for converting the text into audio
And one for playong the audio

"""
import pytesseract
from gtts import gTTS
import pygame
import os
from deep_translator import GoogleTranslator
import speech_recognition as sr

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




def recognize_speech():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()


    with microphone as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print(f"Recognized: {text}")
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

    return None


def translate_text(text, dest_language='es'):
    translated = GoogleTranslator(source='auto', target=dest_language).translate(text)
    print(f"Translated: {translated}")
    return translated

text = recognize_speech()
print(text)
text=translate_text(str(text))
print(text)
audio_file = text_to_speech(text)
play_audio(audio_file)


# Clean up the audio file
if os.path.exists(audio_file):
    os.remove(audio_file)
