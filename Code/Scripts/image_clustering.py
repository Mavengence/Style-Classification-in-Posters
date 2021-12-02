# for loading/processing the images  
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 

# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model
import keras

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import streamlit as st

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import randint
import pandas as pd
import pickle

model_path = "./Models/image_clustering_model.h5"
pca_path = "./Models/pca.pkl"
kmeans_path = "./Models/kmeans.pkl"
corpus_df_path = "./Data/corpus_topics.csv"



def extract_features(file, model):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224,224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img) 
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3) 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features

@st.cache
def predict_centroid(new_img_path):
    model = keras.models.load_model(model_path, compile=False)
    pca_reload = pickle.load(open(pca_path,'rb'))
    kmeans_reload = pickle.load(open(kmeans_path,'rb'))

    feats = extract_features(new_img_path, model)
    feat = feats.reshape(-1,4096)

    x = pca_reload.transform(feat)

    label = kmeans_reload.predict(x)

    return label[0]

def find_similar_images_kmeans(centroid):
    df = pd.read_csv(corpus_df_path)
    same_topic_df = df[df["Centroids"] == centroid]
    return same_topic_df



