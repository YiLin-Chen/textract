'''
higher level class for api
'''

import cv2, os
from google.cloud import vision
from textract.layout.analysis import LayoutAnalyzer
from textract.layout.utils import LayoutType, Page
from textract.model.ocr import TextRecognizer, RecognizerType


class TextExtractor(object):
    def __init__(self, ocr_model = RecognizerType.GOOGLE):
        self._ocr_model = ocr_model
        self._analyzer = LayoutAnalyzer()
        self._recognizer = TextRecognizer(ocr_model)


    def extract(self, image):
        res_str = []
        
        # image = self._analyzer.deskew(image)
        # analyze layout
        page = Page(0, image)
        paragraphs = self._analyzer.extract(page)


        for para in paragraphs:
            if para.get_type() == LayoutType.Paragraph:
                
                # google ocr can accept paragraph as input
                if self._ocr_model == RecognizerType.GOOGLE:
                    ocr_text = self._recognizer.recognize(para.get_src())

                # otherwise split the pararaph into lines
                else:
                    # paragraph to line
                    lines = analyzer.extract(para)
                    ocr_text = ''
                    
                    for l in lines:
                        ocr_text += self._recognizer.recoginze(l.get_src()) + ' '

                    ocr_text = ocr_text[:-1]

    
                res_str.append(ocr_text)
                
        return res_str
