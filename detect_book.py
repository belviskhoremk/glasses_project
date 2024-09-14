import cv2
from PIL import Image
from ultralytics import YOLO
import numpy as np
from paddleocr import PaddleOCR
from gtts import gTTS
import pygame
# from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

openai_api_key = ''
hugging_face_api_key = "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"
# llm_llama = HuggingFaceHub(repo_id = "meta-llama/Meta-Llama-3-8B-Instruct" ,huggingfacehub_api_token = hugging_face_api_key)
llm_gpt4 = ChatOpenAI(model='gpt-4o-mini', openai_api_key=openai_api_key)
# Load the YOLOv8 model
model = YOLO("yolov8l.pt")  # Load a pretrained model (recommended for training)


def detect_and_crop_book(frame, model):
    """
    Function to detect and crop the book from a given frame using YOLOv8 model.
    """
    # Convert the frame to an Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Run the model on the frame
    results = model(frame)  # return a list of Result objects

    # Process results
    for i in range(len(results[0].boxes.cls)):
        if results[0].boxes.cls[i] == 73:  # Class ID for book (change if necessary)
            x_s, y_s, x_f, y_f = map(int, results[0].boxes.xyxy[i])
            cropped_image = image.crop((x_s, y_s, x_f, y_f))
            return cropped_image
    return None


def evaluate_sharpness(frame):
    """
    Function to evaluate the sharpness of a frame using the Laplacian method.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def open_camera_and_detect_book(model):
    """
    Function to open the camera, capture multiple frames, detect the book, and return the cropped book image.
    """
    cap = cv2.VideoCapture(0)  # Open the default camera
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    if not cap.isOpened():
        print('Cannot open camera')
        return None

    result = cv2.VideoWriter('record.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (frame_width, frame_height))

    while True:
        frames = []
        sharpness_scores = []

        for _ in range(10):  # Capture 10 frames
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                sharpness_scores.append(evaluate_sharpness(frame))
                result.write(frame)
                cv2.imshow("Recording webcam", frame)

                if cv2.waitKey(25) == 27:
                    cap.release()
                    result.release()
                    cv2.destroyAllWindows()
                    return None
            else:
                cap.release()
                result.release()
                cv2.destroyAllWindows()
                return None

        # Select the sharpest frame
        best_frame_index = np.argmax(sharpness_scores)
        best_frame = frames[best_frame_index]

        cropped_image = detect_and_crop_book(best_frame, model)
        if cropped_image is not None:
            cap.release()
            result.release()
            cv2.destroyAllWindows()
            return cropped_image


def extract_text_from_image(image):
    """
    Function to extract text from a given image using PaddleOCR.
    """
    ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Load PaddleOCR model
    image_np = np.array(image)
    result = ocr.ocr(image_np, cls=True)

    extracted_text = []

    if result is not None:
        print(result)
        for line in result:
                print("line  " , line)
                for word_info in line:
                    text, confidence = word_info[1]
                    extracted_text.append(text)  # Only append the text, not the confidence

        return extracted_text , True
    else:
        return "we can't extract text from the book , could you order again to take a new photo again" , False


def text_to_speech(text, lang='en'):
    """
    Function to convert text into an audio file.
    """
    tts = gTTS(text=text, lang=lang, tld='com')
    audio_file = 'output.mp3'
    tts.save(audio_file)
    return audio_file


def play_audio(audio_file):
    """
    Function to play the audio file.
    """
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue
    pygame.mixer.quit()


def completed_response(text, llm):
    template = f"""
    You are an AI designed to help with paragraph completion based on the following text:
    Text: {text}

    The above text is extracted from a book picture using OCR. Your task is to complete any cut-off ideas in the extracted text based on the context. If the text is fully extracted and forms a complete idea,
    you don't need to add extra content or don't replace words with other synonym. 
    
    Complete the text:
    """

    chat_template = ChatPromptTemplate.from_template(template)
    chain = chat_template | llm
    response = chain.invoke({"text": text})
    # print("response:", response)
    #
    #
    # extracted_response = response.split("Complete the text:")[-1].strip()
    # print("extracted response:", extracted_response)


    return response.content



def read_book():
    # Run the function to open the camera, detect the book, and extract text
    cropped_image = open_camera_and_detect_book(model)
    if cropped_image:
        extracted_text, status = extract_text_from_image(cropped_image)

        if status:
            full_text = ' '.join(extracted_text)

            print(f"Extracted Text: {full_text}")
            organized_response = completed_response(full_text , llm_gpt4)
            # Convert text to speech and play the audio
            audio_file = text_to_speech(organized_response)
            play_audio(audio_file)
            return True

        else:
            audio_file = text_to_speech(extracted_text)
            play_audio(audio_file)
            print("not word extracted")
            return False


# # Run the main function
# if __name__ == "__main__":
#     main()
