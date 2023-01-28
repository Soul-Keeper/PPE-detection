import cv2
import torch
import streamlit as st

from my_utils import check_center, sqr_of_2_boxes, draw


def process_video(input_path, output_path) -> bool:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/ppe_nano.pt', device=device)
    model.conf = 0.6

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frame = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    progress_bar = st.progress(0)

    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pred = model(frame)
        pred = pred.pandas().xyxy[0]

        persons_data = pred.loc[pred['class'] == 0].copy()
        persons_data["helm"] = None

        ppe_data = pred.loc[pred['class'] != 0].copy()
        ppe_data["used"] = None

        for index_person, row_person in persons_data.iterrows():
            person_box = row_person.to_numpy()[:4].astype(int)
            for index_ppe, row_ppe in ppe_data.iterrows():
                ppe_box = row_ppe.to_numpy()[:4].astype(int)
                if row_ppe['used'] == None:
                    if (sqr_of_2_boxes(ppe_box, person_box) != 0) and (check_center(ppe_box, person_box)):
                        persons_data['helm'] = index_ppe
                        row_ppe['used'] = True

        if not persons_data.empty:
            draw_frame = draw(persons_data, ppe_data, frame)
            out.write(draw_frame)
        else:
            out.write(frame)

        current_frame += 1
        progress_bar.progress(round(current_frame/total_frame * 100))

    cap.release()
    out.release()
    progress_bar.empty()