import cv2
import numpy as np
from network import Fingertips
from hand_detector.detector import YOLO
import time

class circketer():
    def __init__(self, name):
        self.name = name
        self.runs = 0
        self.out = False
        self.shot = 0
        self.ball = 0

    def make_guess(self):
        return np.random.randint(1, 7)

    def pred_num(self, image):
        tl, br = hand.detect(image=image)
        if tl and br is not None:
            cropped_image = image[tl[1]:br[1], tl[0]: br[0]]
            height, width, _ = cropped_image.shape

            prob, pos = fingertips.classify(image=cropped_image)
            pos = np.mean(pos, 0)
            
            prob = np.asarray([(p >= 0.5) * 1.0 for p in prob])
            pred = class_finder(prob)
            return pred

def play_one(inn="hb"):
    preds = []
    image = cam.read()
    for i in range(5):
        ret, image = cam.read()
        p = batsman.pred_num(image)
        if(p == None):
            preds.append(0)
        else:
            preds.append(p)
    human = np.argmax(np.bincount(preds))
    cpu = bowler.make_guess()
    if(inn=="hb"):
        batsman.ball = human
        bowler.ball = cpu
    else:
        batsman.ball = cpu
        bowler.ball = human
    if(human == cpu):
            batsman.out = True
            print("OUT!")
            return num_small[7]
    print("Batsman: ", batsman.ball, "\tBowler: ", bowler.ball)
    batsman.runs += batsman.ball
    return num_small[batsman.ball]


def draw(image, text):
    cv2.circle(image, (image.shape[1]//2, image.shape[0]//2), 48, (127,0,127), -1)
    TEXT = text
    text_size, _ = cv2.getTextSize(TEXT, font, TEXT_SCALE, TEXT_THICKNESS)
    text_origin = (image.shape[1]//2 - text_size[0] // 2, image.shape[0]//2 + text_size[1] // 2)
    cv2.putText(image, TEXT, text_origin, TEXT_FACE, TEXT_SCALE, (127,255,127), TEXT_THICKNESS, cv2.LINE_AA)
    return image

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
        current_class = 0
    return current_class

def plot_prev(image, over, sp=[200, 620]):
    for i in range(len(over)): 
        x = sp[0] + (i*(over[i].shape[1]+10))   
        y = sp[1]                               
        image[y:y+over[i].shape[0], x:x+over[i].shape[1], :] = over[i]
    return image

def human_img(image):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h), (255, 165, 110), 2)
        cv2.putText(img, batsman.name.upper()+" WON!", (x+(w//2)-70, y+(h//2)), TEXT_FACE, 0.7, (255, 165, 110), TEXT_THICKNESS, cv2.LINE_AA)

    cv2.putText(img, batsman.name.upper()+" WON BY "+str(target-batsman.runs)+" RUNS!", (520, 650), TEXT_FACE, 0.7, (255, 165, 110), TEXT_THICKNESS, cv2.LINE_AA)
    return img

TEXT_SCALE = 1.5
TEXT_THICKNESS = 2
TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
hand = YOLO(weights='weights/yolo.h5', threshold=0.8)
fingertips = Fingertips(weights='weights/classes8.h5')

font = cv2.FONT_HERSHEY_COMPLEX
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
res_width, res_height = int(cam.get(3)), int(cam.get(4))
print('Lets Play!')

ctr = 1
flag = 1
num_imgs = []
num_small = []
pos_num = (20, res_width//2+20)
pred_pos = (res_height-70-20, 20)
size_num = (res_width//2-40, res_height-40)

for i in range(0, 9):
    img = cv2.imread("Images/"+str(i)+".png")
    img_big = cv2.resize(img, size_num)
    img_small = cv2.resize(img, (50,70))
    if(i==0 or i==7):
        num_small.append(img_small)
    elif(i==8):
        num_imgs = [img_big] + num_imgs
        num_small.append(img_small)
    else:
        num_imgs.append(img_big)
        num_small.append(img_small)



left = np.zeros((res_height, res_width//2, 3))
trig = 0

batsman = circketer("Dinesh")
bowler = circketer("CPU")
play_one("hb")
batsman = circketer("Dinesh")
bowler = circketer("CPU")
start_time = time.time
target = 0
ball_cnt = 0
over_cnt = 0
over = []
game_state = 1
win_state = 0

while True:
    ret, image = cam.read()
    image[0:left.shape[0], left.shape[1]:, :] = left
    if ret is False:
        break
    val = cv2.waitKey(1)


    if game_state==1:
        if val == 32:
            start_time = time.time()
            trig = 1

        rand_num = num_imgs[bowler.ball]
        image[pos_num[0]:pos_num[0]+rand_num.shape[0], pos_num[1]:pos_num[1]+rand_num.shape[1], :] = rand_num
        cv2.putText(image, batsman.name[:3]+": "+str(batsman.runs)+" in "+str(over_cnt)+"."+str(ball_cnt)+" overs.", (10, 45), TEXT_FACE, 0.7, (180, 50, 0), TEXT_THICKNESS, cv2.LINE_AA)
        try:
            cv2.putText(image, "This over: ", (10, 680), TEXT_FACE, 1, (127,255,127), TEXT_THICKNESS, cv2.LINE_AA)
            image = plot_prev(image, over)
        except:
            pass

        if(trig == 1 and not batsman.out):
            image[pos_num[0]:pos_num[0]+rand_num.shape[0], pos_num[1]:pos_num[1]+rand_num.shape[1], :] = num_imgs[0]
            secs = time.time() - start_time
            if(secs > 3.2):
                over.append(play_one("hb"))
                image[pos_num[0]:pos_num[0]+rand_num.shape[0], pos_num[1]:pos_num[1]+rand_num.shape[1], :] = num_imgs[bowler.ball]
                cv2.imshow("Game Time", image)
                trig = 0
                # Over counter
                ball_cnt += 1
                if(ball_cnt == 6):
                    ball_cnt = 0
                    over_cnt += 1
                if(ball_cnt == 1):
                    over = [over[-1]]
                
            elif(secs >= 2.2 and secs < 3.18):
                image[pos_num[0]:pos_num[0]+rand_num.shape[0], pos_num[1]:pos_num[1]+rand_num.shape[1], :] = num_imgs[0]
                image = draw(image, "1")
            elif(secs >= 1.2 and secs < 2.18):
                image[pos_num[0]:pos_num[0]+rand_num.shape[0], pos_num[1]:pos_num[1]+rand_num.shape[1], :] = num_imgs[np.random.randint(1,7)]
                image = draw(image, "2")
            elif(secs >= 0.2 and secs < 1.18):
                image[pos_num[0]:pos_num[0]+rand_num.shape[0], pos_num[1]:pos_num[1]+rand_num.shape[1], :] = num_imgs[np.random.randint(1,7)]
                image = draw(image, "3")

        if(batsman.out):
            bowler.ball = 8
            batsman.ball = 0
            target = batsman.runs
            print("Target: ", target)
            print("Achieved in ", over_cnt, ".", ball_cnt, " overs")
            batsman.runs = 0
            batsman.out = False
            over = []
            game_state = 1.5
            over_cnt = 0
            ball_cnt = 0

    elif game_state == 1.5:
        rand_num = num_imgs[0]
        image[pos_num[0]:pos_num[0]+rand_num.shape[0], pos_num[1]:pos_num[1]+rand_num.shape[1], :] = rand_num
        image = draw(image, str(target))
        if val == 32:
            game_state = 2

    elif game_state == 2:
        if val == 32:
            start_time = time.time()
            trig = 1
        rand_num = num_imgs[batsman.ball]
        pred_num = num_small[bowler.ball]
        image[pos_num[0]:pos_num[0]+rand_num.shape[0], pos_num[1]:pos_num[1]+rand_num.shape[1], :] = rand_num
        cv2.putText(image, "Computer: "+str(batsman.runs)+" in "+str(over_cnt)+"."+str(ball_cnt)+" overs.", (10, 45), TEXT_FACE, 0.7, (0, 50, 180), TEXT_THICKNESS, cv2.LINE_AA)
        cv2.putText(image, "Target: "+str(target)+". Runs to target: "+str(target-batsman.runs), (670, 45), TEXT_FACE, 0.7, (0, 50, 180), TEXT_THICKNESS, cv2.LINE_AA)
        cv2.putText(image, "Prediction: ", (10, 680), TEXT_FACE, 1, (127,255,127), TEXT_THICKNESS, cv2.LINE_AA)
        image[620:620+pred_num.shape[0], 200:200+pred_num.shape[1], :] = pred_num
        if(trig == 1):
            secs = time.time() - start_time
            image[pos_num[0]:pos_num[0]+rand_num.shape[0], pos_num[1]:pos_num[1]+rand_num.shape[1], :] = num_imgs[np.random.randint(1,7)]
            if(secs > 3.2):
                over.append(play_one("cb"))
                image[pos_num[0]:pos_num[0]+rand_num.shape[0], pos_num[1]:pos_num[1]+rand_num.shape[1], :] = num_imgs[batsman.ball]
                cv2.imshow("Game Time", image)
                trig = 0
                # Over counter
                ball_cnt += 1
                if(ball_cnt == 6):
                    ball_cnt = 0
                    over_cnt += 1
                if(ball_cnt == 1):
                    over = [over[-1]]

            elif(secs >= 2.2 and secs < 3.18):
                image = draw(image, "1")
            elif(secs >= 1.2 and secs < 2.18):
                image = draw(image, "2")
            elif(secs >= 0.2 and secs < 1.18):
                image = draw(image, "3")

        if(batsman.out):
            if(target>batsman.runs):
                print(batsman.name+" wins by "+str(target-batsman.runs)+" runs!")
                win_state = 1
            elif(target<batsman.runs):
                print(bowler.name+" wins!")
                win_state = 2
            else:
                print("Draw!")
                win_state = 0
            game_state = 3

        if(batsman.runs>target):
            print(bowler.name + " wins!")
            game_state = 3
            win_state = 2

        try:
            cv2.putText(image, "This over: ", (670, 680), TEXT_FACE, 1, (127,255,127), TEXT_THICKNESS, cv2.LINE_AA)
            image = plot_prev(image, over, sp=[860, 620])
        except:
            pass

    if game_state == 3:
        if val == 32:
            game_state = 4
        if(win_state == 0):
            rand_num = num_imgs[0]
            image[pos_num[0]:pos_num[0]+rand_num.shape[0], pos_num[1]:pos_num[1]+rand_num.shape[1], :] = rand_num
            cv2.putText(image, "It's a Draw!", (200, 350), TEXT_FACE, 1, (127,255,127), TEXT_THICKNESS, cv2.LINE_AA)
            cv2.putText(image, "It's a Draw!", (840, 350), TEXT_FACE, 1, (127,255,127), TEXT_THICKNESS, cv2.LINE_AA)
        elif(win_state == 1):
            image = human_img(image)
        elif(win_state == 2):
            image = cv2.rectangle(image, (0,0), (1279, 719), (125,0,0), -1)
            cv2.putText(image, "CPU WON!", (530, 350), TEXT_FACE, 1, (0,0,0), TEXT_THICKNESS, cv2.LINE_AA)
        else:
            print("Hi")
    if game_state == 4:
        break
    
    if val == 27:
        break
    cv2.imshow('Game Time', image)

cam.release()
cv2.destroyAllWindows()