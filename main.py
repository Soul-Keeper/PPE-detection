import cv2
import torch
import argparse
import time
import psutil

from pynvml.smi import nvidia_smi

from my_utils import check_center, sqr_of_2_boxes, draw

def main(input_video_path, perfomance_test, save_video):
    if perfomance_test:
        nvsmi = nvidia_smi.getInstance()
        gpu_start = nvsmi.DeviceQuery('memory.free, memory.total')
        ram_start = psutil.virtual_memory()[4]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/ppe_nano.pt', device=device)
    model.conf = 0.6

    cap = cv2.VideoCapture(input_video_path)
    if save_video:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    start_time = time.time()
    num_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        num_frames += 1
        
        pred = model(frame)
        pred = pred.pandas().xyxy[0]

        persons_data = pred.loc[pred['class'] == 0].copy()
        persons_data["helm"] = None

        ppe_data = pred.loc[pred['class'] != 0].copy()
        ppe_data["used"] = None
        
        if num_frames == 300:
            nvsmi = nvidia_smi.getInstance()
            gpu_loop = nvsmi.DeviceQuery('memory.free, memory.total')
            ram_loop = psutil.virtual_memory()[4]

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
            cv2.imshow('test', draw_frame)
            if save_video:
                out.write(draw_frame)
        else:
            cv2.imshow('test', frame)
            if save_video:
                out.write(draw_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    loop_time = time.time() - start_time

    cap.release()
    if save_video:
        out.release()
    cv2.destroyAllWindows()

    if perfomance_test:
        fps = num_frames / loop_time
        gpu_memory_usage = round(float(gpu_start['gpu'][0]['fb_memory_usage']['free']) - float(gpu_loop['gpu'][0]['fb_memory_usage']['free']), 3)
        ram_usage = round((ram_start - ram_loop)/1024/1024, 3)
        print("fps: {} || ram_usage: {} || gpu_memory_usage: {} MiB".format(round(fps, 3), ram_usage, gpu_memory_usage))

parser = argparse.ArgumentParser()
parser.add_argument('--video_path', default='videos/test.mp4')
parser.add_argument('--save_video', default=True)
parser.add_argument('--perfomance_test', default=False)
args = parser.parse_args()

main(args.video_path, args.perfomance_test, args.save_video)