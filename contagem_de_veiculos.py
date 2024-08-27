import numpy as np
import cv2
import sys
import time
import validator
from random import randint

line_in_color = (62, 255, 0)
line_out_color = (0, 0, 255)
bounding_box_color = (255, 128, 0)
tracker_color = (randint(0, 255), randint(0, 255), randint(0, 255))
centroid_color = (randint(0, 255), randint(0, 255), randint(0, 255))
text_color = (randint(0, 255), randint(0, 255), randint(0, 255))
text_position_bgs = (10, 50)
text_position_count_cars = (10, 100)
text_position_count_trucks = (10, 150)
text_size = 1.2
font = cv2.FONT_HERSHEY_SIMPLEX
save_image = True
image_dir = './veiculos'
video_source = '../Traffic.mp4'
video_out = '../result/Result_Traffic.avi'

bgs_types = ['GMG', 'MOG', 'MOG2', 'KNN', 'CNT']
bgs_type = bgs_types[2]

def getKernel(KERNEL_TYPE):
    if KERNEL_TYPE == 'dilation':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if KERNEL_TYPE == 'opening':
        kernel = np.ones((3, 3), np.uint8)
    if KERNEL_TYPE == 'closing':
        kernel = np.ones((3, 3), np.uint8)

    return kernel

def getFilter(img, filter):
    if filter == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, getKernel('closing'), iterations=2)
    if filter == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, getKernel('opening'), iterations=2)
    if filter == 'dilation':
        return cv2.dilate(img, getKernel('dilation'), iterations=2)
    if filter == 'combine':
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, getKernel('closing'), iterations=2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, getKernel('opening'), iterations=2)
        dilation = cv2.dilate(opening, getKernel('dilation'), iterations=2)
        return dilation

def getbgsubtractor(BGS_TYPE):
    if BGS_TYPE == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    if BGS_TYPE == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    if BGS_TYPE == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2(detectShadows=False, varThreshold=100)
    if BGS_TYPE == 'KNN':
        return cv2.createBackgroundSubtractorKNN()
    if BGS_TYPE == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    print('Detector invalido')
    sys.exit(1)

def getCentroid(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return(cx, cy)

def save_frame(frame, file_name, flip=True):
    if flip:
        cv2.imwrite(file_name, np.flip(frame, 2))
    else:
        cv2.imwrite(file_name, frame)

cap = cv2.VideoCapture(video_source)
hasframe, frame = cap.read()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer_video = cv2.VideoWriter(video_out, fourcc, 25, (frame.shape[1], frame.shape[0]), True)

# ROI
bbox = cv2.selectROI(frame, False)
(w1, h1, w2, h2) = bbox

frameArea = h2 * w2
minArea = int(frameArea / 250)
maxArea = 15000

line_in = int(h1)
line_out = int(h2 - 20)

down_limit = int(h1 / 4)

bg_subtractor = getbgsubtractor(bgs_type)

def main():
    frame_number = -1
    cnt_cars = 0
    cnt_trucks = 0
    objects = []
    max_p_age = 2
    pid = 1


    while (cap.isOpened()):

        ok, frame = cap.read()
        if not ok:
            print('Erro')
            break

        roi = frame[h1:h1 + h2, w1:w1 + w2]

        for i in objects:
            i.age_one()

        frame_number += 1
        bg_mask = bg_subtractor.apply(roi)
        bg_mask = getFilter(bg_mask, 'combine')
        (contours, hierarchy) =cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area > minArea and area <= maxArea:
                x, y, w, h = cv2.boundingRect(cnt)
                centroid = getCentroid(x, y, w, h)
                cx = centroid[0]
                cy = centroid[1]
                new = True
                cv2.rectangle(roi, (x, y), (x+50, y-13), tracker_color, -1)
                cv2.putText(roi, 'Car', (x, y -2), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                for i in objects:
                    if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:
                        new = False
                        i.updateCoords(cx, cy)

                        if i.going_DOWN(down_limit) == True:
                            cnt_cars += 1

                            if save_image:
                                save_frame(roi, image_dir + '/car_DOWN_%04d.png' % frame_number)
                                print('ID', i.getId(), 'passou pela estrada em', time.strftime('%c'))
                        break

                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > line_out:
                            i.setDone()

                    if i.timedOut():
                        index = objects.index(i)
                        objects.pop(index)
                        del i

                if new == True:
                    p = validator.MyValidator(pid, cx, cy, max_p_age)
                    objects.append(p)
                    pid += 1

                cv2.circle(roi, (cx, cy), 5, centroid_color, -1)

            elif area >= maxArea:
                x, y, w, h = cv2.boundingRect(cnt)
                centroid = getCentroid(x, y, w, h)
                cx = centroid[0]
                cy = centroid[1]
                new = True
                cv2.rectangle(roi, (x, y), (x+50, y-13), tracker_color, -1)
                cv2.putText(roi, 'Truck', (x, y -2), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                for i in objects:
                    if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:
                        new = False
                        i.updateCoords(cx, cy)

                        if i.going_DOWN(down_limit) == True:
                            cnt_trucks += 1

                            if save_image:
                                save_frame(roi, image_dir + '/car_DOWN_%04d.png' % frame_number)
                                print('ID', i.getId(), 'passou pela estrada em', time.strftime('%c'))
                        break

                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > line_out:
                            i.setDone()

                    if i.timedOut():
                        index = objects.index(i)
                        objects.pop(index)
                        del i

                if new == True:
                    p = validator.MyValidator(pid, cx, cy, max_p_age)
                    objects.append(p)
                    pid += 1

                cv2.circle(roi, (cx, cy), 5, centroid_color, -1)

        for i in objects:
            cv2.putText(roi, str(i.getId()), (i.getX(), i.getY()), font, 0.3, text_color, 1, cv2.LINE_AA)

        str_cars = 'Cars: ' + str(cnt_cars)
        str_trucks = 'Trucks: ' + str(cnt_trucks)

        frame = cv2.line(frame, (w1, line_in), (w1 + w2, line_in), line_in_color, 2)
        frame = cv2.line(frame, (w1, h1 + line_out), (w1 + w2, h1 + line_out), line_out_color, 2)

        cv2.putText(frame, str_cars, text_position_count_cars, font, 1, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, str_cars, text_position_count_cars, font, 1, (232, 162, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, str_trucks, text_position_count_trucks, font, 1, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, str_trucks, text_position_count_trucks, font, 1, (232, 162, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, 'Background Subtractor: ' + bgs_type, text_position_bgs, font, text_size, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, 'Background subtractor: ' + bgs_type, text_position_bgs, font, text_size, (232, 162, 0), 2, cv2.LINE_AA)

        for alpha in np.arange(0.3, 1.1, 0.9)[::-1]:
            overlay = frame.copy()
            output = frame.copy()
            cv2.rectangle(overlay, (w1, h1), (w1 + w2, h1 + h2), bounding_box_color, -1)
            frame = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        cv2.imshow('frame', frame)
        #cv2.imshow('mask', bg_mask)

        writer_video.write(frame)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyWindow()

main()
