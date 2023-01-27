import cv2
import os
import time
import streamlit as st
import torch

from video_processing import process_video

INPUT_PATH = "videos/input/"
OUTPUT_PATH = "videos/output/"

st.title("test")

input_file = st.file_uploader("Upload video", type=[".mp4"])
st.write(input_file.name)

if input_file is not None:
    with open(INPUT_PATH + input_file.name, mode='wb') as w:
        w.write(input_file.getvalue())

    process_video(INPUT_PATH + input_file.name, OUTPUT_PATH + input_file.name)
    os.system('ffmpeg -i ' + OUTPUT_PATH + input_file.name + " -vcodec libx264 " + OUTPUT_PATH + "processed_" + input_file.name)
    video = open(OUTPUT_PATH + "processed_" + input_file.name, 'rb').read()
    st.video(video)
    input_file = None
