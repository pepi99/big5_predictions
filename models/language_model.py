from polyglot.detect import Detector
from polyglot.detect.base import logger as polyglot_logger
polyglot_logger.setLevel("ERROR")

class LanModel:
    def __init__(self):
        self.detector = Detector

    def is_english(self, txt):
        try:
            wrap = self.detector(txt)
            languages = wrap.languages  # The first language will be the most confident language, check if it's English and with more than 98% confidence!
            top_lan = languages[0]
            return top_lan.name == 'English' and top_lan.confidence >= 99
        except Exception as e:
            print('Error with text: ', e)
            # print('Problematic text is: ', txt)
            # print('Exception is: ', e)
            return False


