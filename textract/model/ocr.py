from enum import Enum
from google.cloud import vision
import cv2

class RecognizerType(Enum):
    GOOGLE = 0
    CRNN   = 1

class TextRecognizer(object):
    def __init__(self, model_type):
        if model_type == RecognizerType.GOOGLE:
            self._model = GoogleOCR()

        elif model_type == RecognizerType.CRNN:
            self._model = CRNN()

        else:
            self._model = None

    def recognize(self, src):
        return self._model.predict(src)


class GoogleOCR(object):
    def __init__(self):
        self._client = vision.ImageAnnotatorClient()

    def predict(self, src):
        success, encoded_image = cv2.imencode('.png', src)
        content = encoded_image.tobytes()
        image = vision.types.Image(content=content)
        image_context = vision.types.ImageContext(language_hints=['en'])
        response = self._client.document_text_detection(image=image, image_context=image_context)

        return response.full_text_annotation.text

class CRNN(object):
    def __init__(self):
        pass

    def predict(self, src):
        pass
