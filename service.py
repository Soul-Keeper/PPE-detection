import cv2
import os
import time
import streamlit as st
import torch

from video_processing import process_video

INPUT_PATH = "videos/input/"
OUTPUT_PATH = "videos/output/"

st.title("Personal Protective Equipment detection module")

input_file = st.file_uploader("Upload video", type=[".mp4"])

if input_file is not None:
    st.write("Video is processing, please wait...")
    with open(INPUT_PATH + input_file.name, mode='wb') as w:
        w.write(input_file.getvalue())

    process_video(INPUT_PATH + input_file.name, OUTPUT_PATH + input_file.name)
    st.write("Almost there, just a few seconds...")
    os.system('ffmpeg -y -i ' + OUTPUT_PATH + input_file.name + " -vcodec libx264 " + OUTPUT_PATH + "processed_" + input_file.name)
    video = open(OUTPUT_PATH + "processed_" + input_file.name, 'rb').read()
    st.video(video)
    os.remove(OUTPUT_PATH + input_file.name)
    input_file = None
