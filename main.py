import cv2
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/crowdhuman_nano.pt', device=device)

model.conf = 0.6

cap = cv2.VideoCapture('videos/test.mp4')
ret = True

while ret:
    ret, frame = cap.read()
    
    pred = model(frame)
    pred = pred.pandas().xyxy[0]
    
    for index, row in pred.iterrows():
        color = (0, 0, 255)
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX


        p1 = (int(row['xmin']), int(row['ymin']))
        p2 = (int(row['xmax']), int(row['ymax']))
        draw_frame = cv2.rectangle(frame, p1, p2, color, thickness)

        text = row['name'] + ' ' + str(round(row['confidence'], 2))
        cv2.putText(draw_frame, text, p1, font, 0.5, color, thickness - 1, cv2.LINE_AA)

    cv2.imshow('test', draw_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()