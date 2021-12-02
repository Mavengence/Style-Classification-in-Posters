import easyocr
import os
import pandas as pd
from PIL import Image
import numpy as np
import time
import spacy
from spacy_langdetect import LanguageDetector

import streamlit as st

import de_core_news_sm
import en_core_web_sm

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

import torch
torch.cuda.is_available()

ROTATIONS = [0, 90, 180, 270]

reader = easyocr.Reader(['en', 'de'], gpu=True) 

func = np.vectorize(lambda t: len(t) > 2)


def find_most_likely_words_array(all_words):
    all_words_len = [0, 0, 0, 0]
    
    for i, words in enumerate(all_words):
        
        try:
            long_words = np.array(words)[func(words)]
        except:
            long_words = np.array([])
        
        try:
            for word in long_words:
                try:
                    word = word.strip()
                    lan = detect(word)
                    if lan == "de" or lan == "en" or lan == "fr":
                        all_words_len[i] += 1
                except:
                    pass
        except:
            pass
        
    return np.argmax(all_words_len)

@st.cache
def predict_text(np_img, filename):

    all_words = []
    img = Image.fromarray(np_img)

    for angle in ROTATIONS:
        out = img.rotate(angle)
        out.save(filename)
    
        result = reader.readtext(filename)
        words = [r[1] for r in result]
        all_words.append(words)
        
    index = find_most_likely_words_array(all_words)

    return all_words[index], ROTATIONS[index]