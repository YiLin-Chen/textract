'''
ectractor.py is a module that provides high level api for extracting text
'''

import cv2, os
from google.cloud import vision
from textract.layout.analysis import LayoutAnalyzer
from textract.layout.utils import LayoutType, Page
from textract.model.ocr import TextRecognizer, RecognizerType


class TextExtractor(object):
    '''
    a class used for extracting text from an image
    '''
    def __init__(self, ocr_model = RecognizerType.GOOGLE): 
        self._ocr_model = ocr_model
        self._analyzer = LayoutAnalyzer()
        self._recognizer = TextRecognizer(ocr_model)


    def extract(self, image):
        '''
        Desc: extract text string from an image

        Args:
            - image (numpy.ndarray): target image source

        Returns:
            - res_str (List[str]): a list of paragraph text that is extracted from an image
        '''
       
        res_str = []
        
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
