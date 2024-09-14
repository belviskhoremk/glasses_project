import cv2
from typing import List, Tuple
import numpy as np
import pyttsx3
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from ultralytics import YOLO
from gtts import gTTS
import pygame
from pydub import AudioSegment
import logging
import os
import time
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

hugging_face_api_key = ''
# Initialize TTS engine
engine = pyttsx3.init()

# Load LLaMA model from Hugging Face
text_generator = HuggingFaceHub(repo_id = "meta-llama/Meta-Llama-3-8B-Instruct", huggingfacehub_api_token = hugging_face_api_key, task='text-generation', model_kwargs={"temperature": 0.9, "max_new_token":100})


# Object Detection using YOLO from Ultralytics
def detect_objects(image_path: str) -> List[Tuple[str, Tuple[float, float, float, float]]]:
    """
    Detect objects in the image and return their names and bounding boxes.
    """
    # Run the model on the image
    results = model(image_path,
                    #conf=0.5,
                    max_det=5
                    )

    detected_objects = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        object_name = results[0].names[class_id]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detected_objects.append((object_name, (x1, y1, x2, y2)))

    return detected_objects


def generate_description(objects: List[Tuple[str, Tuple[float, float, float, float]]]) -> str:
    """
    Generate a concise and precise description of the environment based on detected objects and their positions.
    """
    # Format object information with positions
    objects_with_positions = []
    for obj, (x1, y1, x2, y2) in objects:
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # More precise positioning
        if center_x < 0.33:
            x_pos = "left"
        elif center_x < 0.66:
            x_pos = "center"
        else:
            x_pos = "right"

        if center_y < 0.33:
            y_pos = "top"
        elif center_y < 0.66:
            y_pos = "middle"
        else:
            y_pos = "bottom"

        position = f"{y_pos}-{x_pos}" if y_pos != "middle" else x_pos
        objects_with_positions.append(f"{obj} in the {position}")

    # Create a more focused prompt template
    template = """
    Provide a concise description of the environment for a blind person. Focus on:
    1. Main objects and their relative positions
    2. Any potential obstacles or hazards
    3. General layout and space description

    Objects detected: {objects_with_positions}

    Description:
    """

    prompt = PromptTemplate(input_variables=["objects_with_positions"], template=template)
    llm_chain = LLMChain(llm=text_generator, prompt=prompt)
    full_description = llm_chain.run(objects_with_positions=", ".join(objects_with_positions))

    # Extract only the description part
    description = full_description.split("Description:")[1].strip()

    # Post-process the description
    description = description.replace("The environment contains", "There is")
    description = description.replace("There are", "")
    description = description.replace("I can see", "")
    description = ' '.join(description.split())  # Remove extra spaces

    logging.info(f"Generated description: {description}")
    return description


def text_to_speech(text: str, output_file: str = "output.mp3") -> str:
    """
    Convert text to speech and save as an audio file.
    """
    tts = gTTS(text, lang='en', tld='com')
    tts.save(output_file)
    logging.info(f"Text-to-speech audio saved to {output_file}")
    return output_file


def speed_up_audio(file_path: str, speed: float = 1.5, output_file: str = "output_fast.mp3") -> str:
    """
    Speed up the audio file.
    """
    audio = AudioSegment.from_file(file_path)
    faster_audio = audio.speedup(playback_speed=speed)
    faster_audio.export(output_file, format="mp3")
    logging.info(f"Sped-up audio saved to {output_file}")
    return output_file


def play_audio(file_path: str):
    """
    Play the audio file using pygame.
    """
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    logging.info("Finished playing audio")


# def main(image_path: str):
#     """
#     Main function to run the entire pipeline.
#     """
#     try:
#         # 1. Detect objects
#         objects = detect_objects(image_path)
#         logging.info(f"Detected objects: {objects}")
#
#         # 2. Generate environment description
#         description = generate_description(objects)
#
#         # 3. Convert description to audio
#         audio_file = text_to_speech(description, "environment_description.mp3")
#
#         # 4. Speed up the audio (optional)
#         fast_audio_file = speed_up_audio(audio_file, speed=1.3, output_file="environment_description_fast.mp3")
#
#         # 5. Play the audio
#         play_audio(fast_audio_file)
#
#     except Exception as e:
#         logging.error(f"An error occurred: {str(e)}")

def capture_frame(camera_index: int = 2) -> str:
    """
    Capture a frame from the camera and save it as an image file.
    """
    cap = cv2.VideoCapture(camera_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise Exception("Failed to capture frame from camera")

    timestamp = int(time.time())
    filename = f"frame_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    logging.info(f"Captured frame saved as {filename}")
    return filename

def delete_frame(filename: str):
    """
    Delete the captured frame file.
    """
    try:
        os.remove(filename)
        logging.info(f"Deleted frame file: {filename}")
    except Exception as e:
        logging.error(f"Error deleting frame file {filename}: {str(e)}")

def main():
    """
    Main function to run the entire pipeline using camera input.
    """
    try:
        # 1. Capture frame from camera
        frame_filename = capture_frame()

        # 2. Detect objects
        objects = detect_objects(frame_filename)
        logging.info(f"Detected objects: {objects}")

        # 3. Generate environment description
        description = generate_description(objects)

        # 4. Convert description to audio
        audio_file = text_to_speech(description, "environment_description.mp3")

        # 5. Speed up the audio (optional)
        fast_audio_file = speed_up_audio(audio_file, speed=1.3, output_file="environment_description_fast.mp3")

        # 6. Play the audio
        play_audio(fast_audio_file)

        # 7. Clean up - delete the captured frame
        delete_frame(frame_filename)

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

