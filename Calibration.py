import cv2
import numpy as np
from network import Fingertips
from hand_detector.detector import YOLO


def class_finder(prob):
    if np.array_equal(prob, np.array([0, 1, 0, 0, 0])):
        current_class = 1
    elif np.array_equal(prob, np.array([0, 1, 1, 0, 0])):
        current_class = 2
    elif np.array_equal(prob, np.array([0, 1, 1, 1, 0])):
        current_class = 3
    elif np.array_equal(prob, np.array([0, 0, 1, 1, 1])):
        current_class = 3
    elif np.array_equal(prob, np.array([0, 1, 1, 1, 1])):
        current_class = 4
    elif np.array_equal(prob, np.array([1, 1, 1, 1, 1])):
        current_class = 5
    elif np.array_equal(prob, np.array([1, 0, 0, 0, 1])):
        current_class = 6
    else: 
        current_class = -1
    return current_class

hand_detection_method = 'yolo'
hand = YOLO(weights='weights/yolo.h5', threshold=0.8)
fingertips = Fingertips(weights='weights/classes8.h5')
font = cv2.FONT_HERSHEY_COMPLEX

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print('Calibrate Camera')
ctr = 1
flag = 0
num_imgs = []
for i in range(1, 7):
    img = cv2.imread("Images/"+str(i)+".png")
    img = cv2.resize(img, (50,70))
    num_imgs.append(img)

pos_num = (80, 50)

while True:
    ret, image = cam.read()
    if ret is False:
        break

    # hand detection
    if(flag == 0):
        calib_string =  "Hold up the number: " + str(ctr)
        cv2.putText(image, calib_string, (50, 50), font, 1, (49, 238, 219), 2, cv2.LINE_4) 
        ctr_img = num_imgs[ctr-1]
        
        image[pos_num[0]:pos_num[0]+ctr_img.shape[0], pos_num[1]:pos_num[1]+ctr_img.shape[1], :] = ctr_img

    tl, br = hand.detect(image=image)
    if tl and br is not None:
        cropped_image = image[tl[1]:br[1], tl[0]: br[0]]
        height, width, _ = cropped_image.shape

        prob, pos = fingertips.classify(image=cropped_image)
        pos = np.mean(pos, 0)
        
        prob = np.asarray([(p >= 0.5) * 1.0 for p in prob])
        pred = class_finder(prob)
        if(pred == ctr and flag==0):
            ctr += 1
        elif(flag == 1):
            ctr = pred
            cv2.putText(image, "Prediction: " + str(pred), (50, 50), font, 1, (49, 238, 93), 2, cv2.LINE_4) 
            if(1<=pred<=6):
                pred_img = num_imgs[pred-1]
            else:
                pred_img = num_imgs[0]
            image[pos_num[0]:pos_num[0]+pred_img.shape[0], pos_num[1]:pos_num[1]+pred_img.shape[1], :] = pred_img
            
        
        for i in range(0, len(pos), 2):
            pos[i] = pos[i] * width + tl[0]
            pos[i + 1] = pos[i + 1] * height + tl[1]

        index = 0
        color = [(15, 15, 240), (15, 240, 155), (240, 155, 15), (240, 15, 155), (240, 15, 240)]
        for c, p in enumerate(prob):
            if p > 0.5:
                image = cv2.circle(image, (int(pos[index]), int(pos[index + 1])), radius=5,
                                   color=color[c], thickness=-2)
            index = index + 2
    if(ctr == 7):
        print("Calibration Complete!")
        flag = 1
    if cv2.waitKey(1) & 0xff == 27:
        break

    # display image
    cv2.imshow('Camera Calibration', image)

cam.release()
cv2.destroyAllWindows()
