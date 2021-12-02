import pandas as pd
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import shutil



from Code.Scripts.topic_clustering import find_similar_topics, predict_topic, find_similar_images
from Code.Scripts.image_clustering import predict_centroid, find_similar_images_kmeans

def main():

    page = st.sidebar.selectbox("Choose a page", ["Topic Modelling LDA", "Image Clustering"])

    image = Image.open('Imgs/lme.jpg')

    st.sidebar.write("\n\n\n\n")
    show = st.sidebar.image(image, use_column_width=True, caption="Pattern Recognition Lab")

    st.title("Style CLassification in Posters")

    if page == "Topic Modelling LDA":

        st.title("Topic Modelling")
        st.set_option('deprecation.showfileUploaderEncoding', False)

        uploaded_file = st.file_uploader(" ",type=['png', 'jpg', 'jpeg'] )

        if uploaded_file is not None:

            u_img = Image.open(uploaded_file)
            st.sidebar.image(u_img, 'Uploaded Image', use_column_width=True)
 
            user_input = st.text_input(f"Type in the desired Folder Name", "LDA_Similar_Topics")
            predict_button = st.button("Find similar Images by Text and copy them into a Folder")

            if predict_button:

                if uploaded_file is None:
                    st.write("Please upload an Image")

                else:
                    with st.spinner('Finding Images ...'):
                        text, rotate = find_similar_topics(np.array(u_img), uploaded_file.name)
                
                        u_img = u_img.rotate(rotate)

                        joined_text = " ".join(text)

                        topic = predict_topic(joined_text)
                        similar_topic_df = find_similar_images(topic)

                        os.makedirs(user_input)

                        for img_path in similar_topic_df["Path_for_Streamlit"]:
                            shutil.copy(img_path, user_input)

                    show = st.image(u_img, use_column_width=True, caption=f"Sample Plakat - Rotated: {rotate} degrees")
                    st.success(f"Found and Copied {len(similar_topic_df)} Images of the same Topic into the Folder {user_input}!")
                    

    elif page == "Image Clustering":

        st.title("Unsupervised Image Clustering")
        st.title("With PCA and KMeans")
        st.set_option('deprecation.showfileUploaderEncoding', False)

        uploaded_file = st.file_uploader(" ",type=['png', 'jpg', 'jpeg'] )


        if uploaded_file is not None:

            u_img = Image.open(uploaded_file)
            st.sidebar.image(u_img, 'Uploaded Image', use_column_width=True)
 
            user_input = st.text_input(f"Type in the desired Folder Name", "KMeans_Similar_Topics")
            predict_button = st.button("Find similar Images by Image Clustering and copy them into a Folder")

            if predict_button:

                if uploaded_file is None:
                    st.write("Please upload an Image")

                else:
                    with st.spinner('Finding Images ...'):
                        centroid = predict_centroid(uploaded_file.name)

                        similar_topic_df = find_similar_images_kmeans(centroid)

                        os.makedirs(user_input)

                        for img_path in similar_topic_df["Path_for_Streamlit"]:
                            shutil.copy(img_path, user_input)

                    st.success(f"Found and Copied {len(similar_topic_df)} Images of the same Topic into the Folder {user_input}!")


if __name__ == "__main__":
    main()
