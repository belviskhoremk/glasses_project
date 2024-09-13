from text_extract import extract_text
from text_to_speech import text_to_speech, play_audio


def extract_play(path):

    try:

        text = extract_text(path)
        speech = text_to_speech(text)
        play_audio(speech)
        print("successfully played")

    except Exception as e:
        print(e)

if __name__ == '__main__':
    path = "sample-images/sample_book2.jpg"
    extract_play(path)