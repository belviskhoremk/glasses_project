import cv2
from typing import List, Tuple
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from ultralytics import YOLO
from gtts import gTTS
import pygame
from pydub import AudioSegment
import logging
import os
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
from PIL import Image
import time
import requests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

hugging_face_api_key = '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'
arch = 'resnet50'
# Load LLaMA model from Hugging Face
# text_generator = HuggingFaceHub(repo_id = "microsoft/Phi-3.5-mini-instruct", huggingfacehub_api_token = hugging_face_api_key, task='text-generation', model_kwargs={"temperature": 0.9, "max_new_token":100})
from langchain_openai import ChatOpenAI

openai_api_key = 'sk-proj-yCZnQRlv9lPTz4ONFqaOT3BlbkFJUap0kXNVuPLN3rCzFhFf'
text_generator = ChatOpenAI(model='gpt-4o-mini', openai_api_key=openai_api_key)
env_model_file = 'resnet_model/resnet50_places365.pth.tar'


def get_environment(img_path):


    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(env_model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()


    # load the image transformer
    centre_crop = trn.Compose([
            trn.Resize((256,256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load the class label
    file_name = 'resnet_model/categories_places365.txt'
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)


    img = Image.open(img_path)
    input_img = V(centre_crop(img).unsqueeze(0))

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    # Get the highest probability and its corresponding class
    highest_prob = probs[0]
    best_class = classes[idx[0]]

    print('Highest probability: {:.3f} -> {}'.format(highest_prob, best_class))
    return best_class


# Object Detection using YOLO from Ultralytics
def detect_objects(image_path: str) -> List[Tuple[str, Tuple[float, float, float, float]]]:
    """
    Detect objects in the image and return their names and bounding boxes.
    """
    model = YOLO('yolov8l.pt')
    results = model(image_path)

    detected_objects = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        object_name = results[0].names[class_id]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detected_objects.append((object_name, (x1, y1, x2, y2)))

    return detected_objects


def generate_description(objects: List[Tuple[str, Tuple[float, float, float, float]]],environment:str) -> str:
    """
    Generate a concise and precise description of the environment based on detected objects and their positions.
    """
    image_width = 640
    image_height = 480

    objects_with_positions = []

    for obj, (x1, y1, x2, y2) in objects:
        # Calculate the center of the bounding box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Determine the position in terms of quadrants
        if center_x < image_width / 2:
            if center_y < image_height / 2:
                position = "left upper corner"
            else:
                position = "left bottom corner"
        else:
            if center_y < image_height / 2:
                position = "right upper corner"
            else:
                position = "right bottom corner"

        # Append object with position to the list
        objects_with_positions.append(f"{obj} in the {position}")

    # Create a more focused prompt template
    template = """
          You are provided with a set of objects and their positions within a specific environment.
          Please generate a cohesive and concise description that combines only the provided objects and their positions within the {environment}.
          Do not add any external information.
          The description is for a blind person, to allow him to navigate and find things easily. Be really concise and do not make up data.

          Objects detected: {objects_with_positions}

          Environment type: {environment}

          Your task is to create a fluent description that mentions each object along with its position and how it fits into the environment.

          Description:
        """

    # prompt = PromptTemplate(input_variables=["objects_with_positions","environment"], template=template)
    # llm_chain = LLMChain(llm=text_generator, prompt=prompt)
    # full_description = llm_chain.run(objects_with_positions=", ".join(objects_with_positions))
    from langchain_core.prompts import ChatPromptTemplate

    chat_template = ChatPromptTemplate.from_template(template)
    chain = chat_template | text_generator
    full_description = chain.invoke({"objects_with_positions": objects_with_positions, 'environment':environment})
    print(full_description)
    description = full_description.content
    # Extract only the description part
    # description = full_description.split("Description:")[1].strip()

    # Post-process the description
    description = description.replace("The environment contains", "There is")
    description = description.replace("There are", "")
    description = description.replace("I can see", "")
    description = ' '.join(description.split())  # Remove extra spaces

    logging.info(f"Generated description: {description}")
    return description


def text_to_speech(text: str, lang:str, output_file: str = "output.mp3") -> str:
    """
    Convert text to speech and save as an audio file.
    """
    tts = gTTS(text, lang=lang, tld='com')
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


def capture_frame(camera_index: int = 0) -> str:
    """
    Capture a frame from the camera and save it as an image file.
    Includes a warm-up period and a delay before capture to improve frame quality.
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise Exception(f"Failed to open camera with index {camera_index}")

    # Warm-up period: capture and discard frames for 1 second
    warm_up_time = time.time() + 1
    while time.time() < warm_up_time:
        cap.read()

    # Wait for 1 second after warm-up
    time.sleep(1)

    # Capture frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise Exception("Failed to capture frame from camera")

    # Rotate the frame
    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    frame = cv2.resize(frame, (640, 480))
    # Save the frame
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

from deep_translator import GoogleTranslator

def translate_text(text, dest_language='english'):

    translated = GoogleTranslator(source='auto', target=dest_language, ).translate(text)
    print(f"Translated: {translated}")
    return translated

def describe_environment():
    """
    Main function to run the entire pipeline using camera input.
    """
    try:
        dest_language = None
        response = requests.post('http://34.27.59.179/glass/glass_detail/', json={"glass_id": "2"}).json()
        if response:
            # print(response.content)
            dest_language = response['language']
            if dest_language=='french':
                dest_language='fr'
            elif dest_language=='english':
                dest_language='en'
            elif dest_language=='hindi':
                dest_language='hi'
            else:
                dest_language = None
        # 1. Capture frame from camera
        frame_filename = capture_frame()

        # 2. Detect objects
        objects = detect_objects(frame_filename)
        logging.info(f"Detected objects: {objects}")

        #Get the environment type
        environment = get_environment(frame_filename) or "unknown environment"
        logging.info(f"Environment type: {environment}")

        # 3. Generate environment description
        description = generate_description(objects,environment)
        if dest_language is not None:
            description = translate_text(description, dest_language=dest_language)
        else:
            description = translate_text(description)
        # 4. Convert description to audio
        audio_file = text_to_speech(description, dest_language, "environment_description.mp3")

        # 5. Speed up the audio (optional)
        # fast_audio_file = speed_up_audio(audio_file, speed=1.3, output_file="environment_description_fast.mp3")

        # 6. Play the audio
        play_audio(audio_file)

        # 7. Clean up - delete the captured frame
        delete_frame(frame_filename)
        os.remove(audio_file)

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")



if __name__ == "__main__":
    describe_environment()

