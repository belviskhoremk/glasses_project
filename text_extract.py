import cv2
from PIL import Image
# from pytesseract import pytesseract


def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh

from paddleocr import PaddleOCR, draw_ocr

def pdp(img_path):
    ocr = PaddleOCR(use_angle_cls=True, lang='en')  # need to run only once to download and load model into memory


    # Extract text
    result = ocr.ocr(img_path, cls=True)
    extracted_text = []
    for line in result:
        for word_info in line:
            text, confidence = word_info[1]
            extracted_text.append((text, confidence))

    for text, confidence in extracted_text:
        print(f"Text: {text}, Confidence: {confidence}")

def extract_text(path , lang ="eng"):
    """
    function to extract texts from an image of given path
    :param path:
    :param lang:preferernce language
    :return -> str:extracted text form a given image
    """
    pass
    image = cv2.imread(path)

    # Convert the processed image to PIL format
    img_pil = Image.fromarray(image)

    # Extract text using Tessersact OCR
    text = pytesseract.image_to_string(img_pil, lang=lang)

    print(text)
    print("\n\ntype", type(text))

    return text

if __name__ == '__main__':
    path = 'sample-images/sample_book4.jpg'
    pdp(path)