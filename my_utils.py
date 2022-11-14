import cv2

#params for drawing boxes
GOOD_COLOR = (100, 255, 100)
BAD_COLOR = (0, 0, 255)

#params for putting text
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
TEXT_COLOR = (255, 255, 0)
THICKNESS = 1

def check_center(boxA, boxB): #ppe, person
    xc = (boxA[2] + boxA[0]) / 2
    yc = (boxA[3] + boxA[1]) / 2
    trashold = (boxB[3] - boxB[1]) * 0.6
    if (boxB[0] < xc < boxB[2]) and (boxB[1] < yc < (boxB[3] - trashold)):
        return True

def sqr_of_2_boxes(boxA, boxB):
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    
    return float(boxAArea / boxBArea)

def draw(persons_data, ppe_data, frame): 
    for index_person, row_person in persons_data.iterrows():
        if row_person['helm'] != None:
            p1 = (int(row_person['xmin']), int(row_person['ymin']))
            p2 = (int(row_person['xmax']), int(row_person['ymax']))
            draw_frame = cv2.rectangle(frame, p1, p2, color=GOOD_COLOR, thickness=2)

            text_point = (int(row_person['xmax']) + 5, int(row_person['ymin']) + 10)
            text = "PERSON " + str(round(row_person['confidence'], 2))
            draw_frame = cv2.putText(draw_frame, text, text_point, FONT, FONT_SCALE, TEXT_COLOR, THICKNESS, cv2.LINE_AA)

            ppe_info = ppe_data.loc[[row_person['helm']]]
            p1 = (int(ppe_info['xmin']), int(ppe_info['ymin']))
            p2 = (int(ppe_info['xmax']), int(ppe_info['ymax']))
            draw_frame = cv2.rectangle(draw_frame, p1, p2, color=(255, 255, 255), thickness=1)

            text_point = (int(row_person['xmax']) + 5, int(row_person['ymin']) + 25)
            text = "HELM " + str(round(list(ppe_info['confidence'])[0], 2))
            draw_frame = cv2.putText(draw_frame, text, text_point, FONT, FONT_SCALE, TEXT_COLOR, THICKNESS, cv2.LINE_AA)

        if row_person['helm'] == None:
            p1 = (int(row_person['xmin']), int(row_person['ymin']))
            p2 = (int(row_person['xmax']), int(row_person['ymax']))
            draw_frame = cv2.rectangle(frame, p1, p2, color=BAD_COLOR, thickness=2)

            text_point = (int(row_person['xmax']) + 5, int(row_person['ymin']) + 10)
            text = "PERSON " + str(round(row_person['confidence'], 2))
            draw_frame = cv2.putText(draw_frame, text, text_point, FONT, FONT_SCALE, TEXT_COLOR, THICKNESS, cv2.LINE_AA)
   
    return draw_frame
