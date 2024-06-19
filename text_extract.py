from PIL import Image
from pytesseract import pytesseract


def extract_text(path , lang ="eng"):
    """
    function to extract texts from an image of given path
    :param path:
    :param lang:preferernce language
    :return -> str:extracted text form a given image
    """
    img  = Image.open(path)

    text = pytesseract.image_to_string(img, lang=lang)

    print(text)
    print("\n\ntype", type(text))

    return text

# if __name__ == '__main__':
#     path = 'sample-images/sample_book2.jpg'
#     extract_text(path)