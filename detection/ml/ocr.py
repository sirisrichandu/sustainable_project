import easyocr
import cv2

reader = easyocr.Reader(['en'], gpu=False)

def extract_text(image_path):
    image = cv2.imread(image_path)
    results = reader.readtext(image)

    texts = []
    for _, text, conf in results:
        if conf > 0.4:
            texts.append(text)

    return " ".join(texts)
