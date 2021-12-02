import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import spacy
from spacy_langdetect import LanguageDetector
import easyocr
import de_core_news_sm
import en_core_web_sm
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

import re
import os
import string
import sys
from pathlib import Path
import pickle

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

import gensim
import gensim.corpora as corpora
from pprint import pprint

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import streamlit as st
#pyLDAvis.enable_notebook()

module_path = str(Path.cwd().parents[0] / "Scripts")
if module_path not in sys.path:
    sys.path.append(module_path)
    
import warnings
warnings.filterwarnings('ignore')

from Code.Scripts.easyocr import predict_text
from Code.Scripts.notebook_scripts import split_array, stop_word_removal


corpus_df_path = "./Data/corpus_topics.csv"
model_path = "./Models/topic_model.sav"

@st.cache
def predict_topic(text):
    loaded_model = pickle.load(open(model_path, 'rb'))
    result = loaded_model.transform(text)
    return result[0][0]

def find_similar_topics(img, filename):
    text, rotate = predict_text(img, filename)
    return text, rotate

def find_similar_images(topic):
    df = pd.read_csv(corpus_df_path)
    same_topic_df = df[df["Topic"] == topic]
    return same_topic_df