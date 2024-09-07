import cv2
import numpy as np
import pyttsx3
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from ultralytics import YOLO
from gtts import gTTS
import pygame
from pydub import AudioSegment
import os

hugging_face_api_key = ''
# Initialize TTS engine
engine = pyttsx3.init()

# Load LLaMA model from Hugging Face
text_generator = HuggingFaceHub(repo_id = "meta-llama/Meta-Llama-3-8B-Instruct", huggingfacehub_api_token = hugging_face_api_key, task='text-generation')


# Object Detection using YOLO from Ultralytics
def detect_objects(image_path):
    model = YOLO('yolov8l.pt')  # Load YOLOv8 model
    results = model(image_path)  # Run the model on the image

    detected_objects = []

    # Iterate over detected boxes in results[0]
    for box in results[0].boxes:  # boxes object contains detected data
        class_id = int(box.cls[0])  # Get the class id for the detected object
        detected_objects.append(results[0].names[class_id])  # Map class_id to object name

    return detected_objects

# Generate environment description using HuggingFaceHub LLM
def generate_description(objects):
    # hf_llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.3", model_kwargs={"temperature": 0.6})
    template = PromptTemplate(input_variables=["objects"], template="The environment contains the following objects: {objects}. Please describe the environment.")
    llm_chain = LLMChain(llm=text_generator, prompt=template)
    description = llm_chain.run(objects=", ".join(objects))
    return description

# Text-to-speech conversion using gTTS
def text_to_speech(text, output_file="output.mp3"):
    tts = gTTS(text, lang='en', tld='com')
    tts.save(output_file)

def speed_up_audio(file_path, speed=1.5, output_file="output_fast.mp3"):
    audio = AudioSegment.from_file(file_path)
    faster_audio = audio.speedup(playback_speed=speed)
    faster_audio.export(output_file, format="mp3")
    return output_file

# Play the audio using pygame
def play_audio(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

# Full pipeline
def main(image_path):
    # 1. Detect objects
    objects = detect_objects(image_path)

    # 2. Generate environment description
    description = generate_description(objects)

    # 3. Convert description to audio
    text_to_speech(description, "environment_description.mp3")

    fast_file = speed_up_audio('environment_description.mp3', speed=1.5, output_file="environment_description_fast.mp3")

    # 4. Play the audio
    play_audio(fast_file)

# Example usage

if __name__ == "__main__":
    main("/home/belvisk/Downloads/images.jpeg")
