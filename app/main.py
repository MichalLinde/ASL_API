import os
import numpy as np
import cv2
import mediapipe as mp
from tensorflow import keras
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
from pydantic import BaseModel
from tempfile import NamedTemporaryFile

import time
from numpy import zeros
from matplotlib import pyplot as plt




class Item(BaseModel):
    link: str

app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers
)

# FUNCTIONS WORKING WITH MP HOLISTIC MODEL

# Initialazing mp holistic for keypoints
mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


# Finding landmarks with model holistic
def mediapipe_detection(image, model):
    # creating contrast on img
    '''
    lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # applying CLAHE to L-channel   - feel free to try different values for the limit and grid size
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl,a,b)) # merge the CLAHE enhanced L-channel with the a and b channel
    image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR) #onverting image from LAB Color model to BGR color spcae
    '''
    # detections
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # color conversion BGR 2 RGB
    image.flags.writeable = False  # image is no longer writeable
    results = model.process(image)  # make prediction
    image.flags.writeable = True  # image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # color conversion RGB 2 BGR
    return image, results


# Drawing landmarks
def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )  # draw pose connections

    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )  # draw left hand connections

    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )  # draw right hand connections


# Preparing vector of date
def extract_keypoints(results):
    if (results.pose_landmarks == None):  # if there is no landmarks
        return [], [], [], [], [],

    else:
        # face
        x = 0
        y = 0
        for i in range(11):
            if results.pose_landmarks.landmark[i].visibility > 0.95:
                x += results.pose_landmarks.landmark[i].x
                y += results.pose_landmarks.landmark[i].y
            else:
                return [], [], [], [], [],
        face = np.array([x / 11, y / 11])  # the middle point from all points on face

        # pose
        pose_left = []
        pose_right = []

        if (results.pose_landmarks.landmark[11].visibility > 0.80 and results.pose_landmarks.landmark[
            12].visibility > 0.80):  # middle point between choulders
            pose_left.append(
                ((results.pose_landmarks.landmark[11].x + results.pose_landmarks.landmark[12].x) / 2) - face[0])
            pose_left.append(
                ((results.pose_landmarks.landmark[11].y + results.pose_landmarks.landmark[12].y) / 2) - face[1])
            pose_right.append(
                ((results.pose_landmarks.landmark[11].x + results.pose_landmarks.landmark[12].x) / 2) - face[0])
            pose_right.append(
                ((results.pose_landmarks.landmark[11].y + results.pose_landmarks.landmark[12].y) / 2) - face[1])
        else:
            pose_left.append(0)
            pose_left.append(0)
            pose_right.append(0)
            pose_right.append(0)

        for i in range(11, 17):  # points from pose - arms and shoulder
            if (results.pose_landmarks.landmark[i].visibility > 0.80):
                if (i % 2 == 0):
                    pose_right.append(-1 * (results.pose_landmarks.landmark[i].x - face[0]))
                    pose_right.append(results.pose_landmarks.landmark[i].y - face[1])
                else:
                    pose_left.append(results.pose_landmarks.landmark[i].x - face[0])
                    pose_left.append(results.pose_landmarks.landmark[i].y - face[1])
            else:
                if (i % 2 == 0):
                    pose_right.append(0)
                    pose_right.append(0)
                else:
                    pose_left.append(0)
                    pose_left.append(0)

        pose_left = np.array(pose_left)
        pose_right = np.array(pose_right)

        # right hand
        if (results.right_hand_landmarks == None):
            right_hand = []
        else:
            right_hand = np.array(
                [[-1 * (res.x - face[0]), res.y - face[1]] for res in results.right_hand_landmarks.landmark]).flatten()

        # left hand
        if (results.left_hand_landmarks == None):
            left_hand = []
        else:
            left_hand = np.array(
                [[res.x - face[0], res.y - face[1]] for res in results.left_hand_landmarks.landmark]).flatten()

        return face, pose_left, pose_right, left_hand, right_hand


# FUNCTIONS FOR extract_sequence

def init_global_variables_on_zero(all_variables=True):
    global frame_num, all_keypoints, left_zeros_fixed, right_zeros_fixed
    global number_of_blank_pose_left, number_of_blank_pose_right, number_of_blank_left_hand, number_of_blank_right_hand
    global vector_of_blank_pose_left, vector_of_blank_pose_right, vector_of_blank_left_hand, vector_of_blank_right_hand
    global all_frames_pose_left, all_frames_pose_right, all_frames_left_hand, all_frames_right_hand

    number_of_blank_pose_left = 0
    number_of_blank_pose_right = 0
    number_of_blank_left_hand = 0
    number_of_blank_right_hand = 0

    vector_of_blank_pose_left = []
    vector_of_blank_pose_right = []
    vector_of_blank_left_hand = []
    vector_of_blank_right_hand = []

    if all_variables:
        all_frames_pose_left = []
        all_frames_pose_right = []
        all_frames_left_hand = []
        all_frames_right_hand = []

        frame_num = 1
        all_keypoints = []
        left_zeros_fixed = False
        right_zeros_fixed = False


def counting_number_of_blanks():
    global frame_num, all_keypoints, left_zeros_fixed, right_zeros_fixed
    global number_of_blank_pose_left, number_of_blank_pose_right, number_of_blank_left_hand, number_of_blank_right_hand
    global vector_of_blank_pose_left, vector_of_blank_pose_right, vector_of_blank_left_hand, vector_of_blank_right_hand
    global all_frames_pose_left, all_frames_pose_right, all_frames_left_hand, all_frames_right_hand
    global pose_left, pose_right, left_hand, right_hand

    # checking existence of landmarks on body and hands
    if len(left_hand) == 0:
        left_hand = np.zeros(42)
        vector_of_blank_left_hand.append(1)
    else:
        vector_of_blank_left_hand.append(0)

    if len(right_hand) == 0:
        right_hand = np.zeros(42)
        vector_of_blank_right_hand.append(1)
    else:
        vector_of_blank_right_hand.append(0)

    if len(pose_left) - np.count_nonzero(pose_left) != 0:
        vector_of_blank_pose_left.append(1)
    else:
        vector_of_blank_pose_left.append(0)

    if len(pose_right) - np.count_nonzero(pose_right) != 0:
        vector_of_blank_pose_right.append(1)
    else:
        vector_of_blank_pose_right.append(0)

    # number of blanks in last 12 frames not including last one
    if (frame_num > sequence_length):
        vector_of_blank_pose_left = vector_of_blank_pose_left[-31:]
        vector_of_blank_pose_right = vector_of_blank_pose_right[-31:]
        vector_of_blank_left_hand = vector_of_blank_left_hand[-31:]
        vector_of_blank_right_hand = vector_of_blank_right_hand[-31:]

    number_of_blank_pose_left = np.count_nonzero(vector_of_blank_pose_left[-31:-1])
    number_of_blank_pose_right = np.count_nonzero(vector_of_blank_pose_right[-31:-1])
    number_of_blank_left_hand = np.count_nonzero(vector_of_blank_left_hand[-31:-1])
    number_of_blank_right_hand = np.count_nonzero(vector_of_blank_right_hand[-31:-1])


# Updating hole story for single part
def update_all_frames():
    global frame_num, all_keypoints, left_zeros_fixed, right_zeros_fixed
    global number_of_blank_pose_left, number_of_blank_pose_right, number_of_blank_left_hand, number_of_blank_right_hand
    global vector_of_blank_pose_left, vector_of_blank_pose_right, vector_of_blank_left_hand, vector_of_blank_right_hand
    global all_frames_pose_left, all_frames_pose_right, all_frames_left_hand, all_frames_right_hand
    global pose_left, pose_right, left_hand, right_hand

    all_frames_pose_left.append(pose_left)
    all_frames_pose_right.append(pose_right)
    all_frames_left_hand.append(left_hand)
    all_frames_right_hand.append(right_hand)

    # extracting last 13 frames
    if (frame_num > sequence_length):
        all_frames_pose_left = all_frames_pose_left[-31:]
        all_frames_pose_right = all_frames_pose_right[-31:]
        all_frames_left_hand = all_frames_left_hand[-31:]
        all_frames_right_hand = all_frames_right_hand[-31:]


def fixing_zeros_and_counting_varations(all_frames_part, check_varation=True, fix_zeros=True):
    global frame_num, all_keypoints, left_zeros_fixed, right_zeros_fixed
    global number_of_blank_pose_left, number_of_blank_pose_right, number_of_blank_left_hand, number_of_blank_right_hand
    global all_frames_pose_left, all_frames_pose_right, all_frames_left_hand, all_frames_right_hand
    global pose_left, pose_right, left_hand, right_hand

    varations = []
    col = []
    # creating vector of colmun
    for it in range(len(all_frames_part[0])):
        for frame in all_frames_part:
            col.append(frame[it])

        # fixing zeros in whole 12 frame sequance
        if fix_zeros:
            for it2 in range(1, len(col) - 1):
                if (col[it2] == 0 and col[it2 + 1] == 0):
                    col[it2] = col[it2 - 1]
                    all_frames_part[it2][it] = col[it2 - 1]
                if (col[it2] == 0):
                    col[it2] = (col[it2 - 1] + col[it2 + 1]) / 2
                    all_frames_part[it2][it] = col[it2]
        # fixing zeros only on the end of sequence
        else:
            if (col[29] == 0 and col[30] == 0):
                col[29] = col[28]
                all_frames_part[29][it] = col[28]
            if (col[29] == 0):
                col[29] = (col[28] + col[30]) / 2
                all_frames_part[29][it] = col[29]

        # variation from every colmun
        if check_varation:
            # wyliczamy wariancje
            col = 10 * np.array(col[:30])  # *10 working good for next if with treshold
            srednio = sum(col) / sequence_length
            varations.append(np.sum((np.array(col) - srednio) ** 2) / sequence_length)
        col = []

    # counting varation and return if check varation
    if check_varation:
        if (np.sum(varations) >= 0.8):
            return [1, 1]
        else:
            return [1, 404]

    # return for fixiing and not fixing zeros
    return [1, 404]


# LOGIC

def extract_sequence(results):
    global frame_num, all_keypoints, left_zeros_fixed, right_zeros_fixed
    global number_of_blank_pose_left, number_of_blank_pose_right, number_of_blank_left_hand, number_of_blank_right_hand
    global vector_of_blank_pose_left, vector_of_blank_pose_right, vector_of_blank_left_hand, vector_of_blank_right_hand
    global all_frames_pose_left, all_frames_pose_right, all_frames_left_hand, all_frames_right_hand
    global pose_left, pose_right, left_hand, right_hand

    # extracting keypoints
    face, pose_left, pose_right, left_hand, right_hand = extract_keypoints(results)

    if len(face) == 0:  # if there is no landmarks
        print("there is nothing on sreen")
        init_global_variables_on_zero()
        return 404

    # checking how many blanks is in every single part of body in last 12 frames
    counting_number_of_blanks()

    # updating whole story for single part
    update_all_frames()

    if (frame_num < sequence_length + 1):
        frame_num += 1
        return 404

    else:

        lh = [404, 404]
        lp = [404, 404]
        ph = [404, 404]
        pp = [404, 404]
        # left hand and arm cant have more then 2 blanks frame and first analizng frame cant contain only zeros
        if (number_of_blank_left_hand <= 5 and number_of_blank_pose_left <= 5 and all_frames_left_hand[0][
            0] != 0 and len(pose_left) - np.count_nonzero(all_frames_pose_left[0]) == 0):
            if (left_zeros_fixed):
                lh = fixing_zeros_and_counting_varations(all_frames_left_hand, True, False)
                lp = fixing_zeros_and_counting_varations(all_frames_pose_left, False, False)
            else:
                lh = fixing_zeros_and_counting_varations(all_frames_left_hand)
                lp = fixing_zeros_and_counting_varations(all_frames_pose_left, False)

        # right hand and arm cant have more then 2 blanks frame and first analizng frame cant contain only zeros
        if (number_of_blank_right_hand <= 6 and number_of_blank_pose_right <= 6 and all_frames_right_hand[0][
            0] != 0 and len(pose_right) - np.count_nonzero(all_frames_pose_right[0]) == 0):
            if (right_zeros_fixed):
                ph = fixing_zeros_and_counting_varations(all_frames_right_hand, True, False)
                pp = fixing_zeros_and_counting_varations(all_frames_pose_right, False, False)
            else:
                ph = fixing_zeros_and_counting_varations(all_frames_right_hand)
                pp = fixing_zeros_and_counting_varations(all_frames_pose_right, False)

        # not good data for prediction: 2 blanks next to eachother or too much blanks frame
        if (lh[0] == 404 or lp[0] == 404):
            left_zeros_fixed = False
        else:
            left_zeros_fixed = True
        if (ph[0] == 404 or pp[0] == 404):
            right_zeros_fixed = False
        else:
            right_zeros_fixed = True
        if (left_zeros_fixed == False and right_zeros_fixed == False):
            print("coutch not enough landmarks ")
            return 404

        # 2 hands are moving
        elif (lh[1] == 1 and ph[1] == 1):
            print(
                "do sieci na 2 rece")  # TO DO ... jak bd sie jebac moge polaczyc pose'y z soba i wyjebac jeden wspolny pnkt#####
            for i in range(sequence_length):
                keypoints = np.concatenate([all_frames_pose_left[i], all_frames_left_hand[i], all_frames_pose_right[i],
                                            all_frames_right_hand[i]])
                all_keypoints.append(keypoints)
            print()
            return 2

            # 1 hand is moving
        elif (lh[1] == 1 or ph[1] == 1):
            print("do sieci dla 1 reki")
            if (lh[1] == 1):
                for i in range(sequence_length):
                    keypoints = np.concatenate([all_frames_pose_left[i], all_frames_left_hand[i]])
                    all_keypoints.append(keypoints)
            else:
                for i in range(sequence_length):
                    keypoints = np.concatenate([all_frames_pose_right[i], all_frames_right_hand[i]])
                    all_keypoints.append(keypoints)
            print()
            return 1

        # no move detected
        else:
            print("do sieci dla nie ruchomych rak")
            all_keypoints = np.concatenate(
                [sum(all_frames_pose_left) / sequence_length, sum(all_frames_left_hand) / sequence_length,
                 sum(all_frames_pose_right) / sequence_length, sum(all_frames_right_hand) / sequence_length])
            print()
            return 0






# folders info
no_sequences = 50
sequence_length = 30
DATA_PATH = os.path.join('MP_Data')

# bez rak
actions0 = np.array(['hurt', 'eat', 'I_me'])
colors0 = [(255, 153, 51) ,(245,117,16), (117,245,16)]

#1reka
#actions1 = np.array(['HE', 'Thanks', 'Pleas', 'is', 'diabetic', 'DRINK', 'head','Thirsty', 'broke', 'arm', 'hello','need', 'hungry'])
#actions1 = np.array(['HE', 'need', 'is', 'Thirsty', 'head', 'diabetic', 'Pleas', 'Thanks', 'hello', 'hungry', 'arm', 'broke', 'DRINK'])
actions1 = np.array(['Arm', 'Call', 'Dizzy', 'Drink', 'Fall_down', 'Hello', 'Infection', 'Need', 'Please', 'Thirsty'])
colors1 = [(255, 153, 51) ,(245,117,16), (117,245,16), (16,117,245), (245,117,16), (117,245,16), (16,117,245), (255, 153, 51) ,(245,117,16), (117,245,16)]

#2reka
#actions2 = np.array(['cold(2)', 'hit(2)', 'IN', 'Want(2)'])
#actions2 = np.array(['Want(2)', 'IN', 'hit(2)', 'cold(2)'])
actions2 = np.array(['Alergy', 'Hit', 'Hospital', 'In', 'Thanks', 'Want'])
colors2 = [(255, 153, 51) ,(245,117,16), (117,245,16), (16,117,245), (245,117,16), (117,245,16)]



# importing model
#model0 = keras.models.load_model("moje_basic_0_1_2/action2d_0hand.h5")
#model1 = keras.models.load_model("moje_basic_0_1_2/action2d_1hand_94.h5")
#model2 = keras.models.load_model("moje_basic_0_1_2/action2d_2hand_99.h5")

model0 = keras.models.load_model("app/model_convolucial_0hand.h5")
model1 = keras.models.load_model("app/model_bidirectional_droput_final_1hand.h5")
model2 = keras.models.load_model("app/model_bidirectional_droput_final_2hand.h5")










def processVideo(video):
    init_global_variables_on_zero()
    global frame_num, all_keypoints
    global number_of_blank_pose_left, number_of_blank_pose_right, number_of_blank_left_hand, number_of_blank_right_hand
    global vector_of_blank_pose_left, vector_of_blank_pose_right, vector_of_blank_left_hand, vector_of_blank_right_hand
    global all_frames_pose_left, all_frames_pose_right, all_frames_left_hand, all_frames_right_hand
    global pose_left, pose_right, left_hand, right_hand
    print(1)
    sentence = []
    sentence.append("no_landmarks")
    clean = True
    last_one = ""
    last_one_counter = 0
    threshold = 0.6


    cap = cv2.VideoCapture(video)
    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.4, min_tracking_confidence=0.4) as holistic:
        print(2)
        while cap.isOpened():
            print(3)
            # Read feed
            ret, frame = cap.read()
            if not ret:
                print("EOF")
                break
            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            # Draw landmarks
            # extract sequence
            network_choose = extract_sequence(results)

            # choosing type of network
            if network_choose == 0:

                # making prediction
                res = model0.predict(np.expand_dims(all_keypoints, axis=0))[0]
                print(actions0[np.argmax(res)])

                # collecting 5 last words for top bar
                if res[np.argmax(res)] > threshold:

                    # last 5 frames must predict the same sign

                    if (last_one == actions0[np.argmax(res)]):
                        last_one_counter += 1
                    else:
                        last_one = actions0[np.argmax(res)]
                        last_one_counter = 1

                    if (last_one_counter > 4):
                        # no_landmarks deleting
                        if (clean):
                            sentence = []
                            clean = False

                        if len(sentence) > 0:
                            if actions0[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions0[np.argmax(res)])
                        else:
                            sentence.append(actions0[np.argmax(res)])
                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # showing probabilities for every word
                for num, prob in enumerate(res):
                    cv2.rectangle(image, (0, 60 + num * 30), (int(prob * 100), 90 + num * 30), colors0[num], -1)
                    cv2.putText(image, actions0[num], (0, 85 + num * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                                2, cv2.LINE_AA)

            elif network_choose == 1:

                # making prediction
                res = model1.predict(np.expand_dims(all_keypoints, axis=0))[0]
                print(actions1[np.argmax(res)])

                # collecting 5 last words for top bar
                if res[np.argmax(res)] > threshold:

                    # last 5 frames must predict the same sign

                    if (last_one == actions1[np.argmax(res)]):
                        last_one_counter += 1
                    else:
                        last_one = actions1[np.argmax(res)]
                        last_one_counter = 1

                    if (last_one_counter > 4):
                        # no_landmarks deleting
                        if (clean):
                            sentence = []
                            clean = False

                        if len(sentence) > 0:
                            if actions1[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions1[np.argmax(res)])
                        else:
                            sentence.append(actions1[np.argmax(res)])
                if len(sentence) > 6:
                    sentence = sentence[-5:]

                # showing probabilities for every word
                for num, prob in enumerate(res):
                    cv2.rectangle(image, (0, 60 + num * 30), (int(prob * 100), 90 + num * 30), colors1[num], -1)
                    cv2.putText(image, actions1[num], (0, 85 + num * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                                2, cv2.LINE_AA)

            elif network_choose == 2:

                # making prediction
                res = model2.predict(np.expand_dims(all_keypoints, axis=0))[0]
                print(actions2[np.argmax(res)])

                # collecting 5 last words for top bar
                if res[np.argmax(res)] > threshold:

                    # last 5 frames must predict the same sign

                    if (last_one == actions2[np.argmax(res)]):
                        last_one_counter += 1
                    else:
                        last_one = actions2[np.argmax(res)]
                        last_one_counter = 1

                    if (last_one_counter > 4):
                        # no_landmarks deleting
                        if (clean):
                            sentence = []
                            clean = False

                        if len(sentence) > 0:
                            if actions2[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions2[np.argmax(res)])
                        else:
                            sentence.append(actions2[np.argmax(res)])
                if len(sentence) > 5:
                    sentence = sentence[-5:]

            # needed because of np concatenation
            all_keypoints = []


    cap.release()
    return {"message": " ".join(sentence)}


@app.get("/")
async def root():
    return {"message": "Online!"}

@app.post("/video/test-asl")
def testASL(file: UploadFile = File(...)):
    temp = NamedTemporaryFile(delete=False)
    try:
        try:
            contents = file.file.read()
            with temp as f:
                f.write(contents)
        except Exception:
            return {"message": "There was an error uploading the file"}
        finally:
            file.file.close()

        res = processVideo(temp.name)
    except Exception:
        return {"message": "There was an error processing the file"}
    finally:
        os.remove(temp.name)

    return res


@app.get("/{full_path:path}")
def predictASL(full_path: str):
    res = processVideo(full_path)

    return res

if __name__ == "__main__":
    init_global_variables_on_zero()
    port = int(os.environ.get("PORT", 5050))
    run(app, host="0.0.0.0", port=port)


# INSTRUKCJA URUCHOMIENIA API
#
# W PRZEGLĄDARCE:
# 0.0.0.0:5050/ - STATUS CHECK
# 0.0.0.0:5050/{LINK} - PREDYKCJA
# 0.0.0.0:5050/docs - INTERAKTYWNA INSTRUKCJA API
#
# NA TELEFONIE:
# SPRAWDZIĆ ADRES IP SIECI
# W PRZEGLĄDARCE NA TELEFONIE ODPALIĆ {ADRES IP SIECI}:5050/
#
# PRZYKŁADOWY LINK:
# http://0.0.0.0:5050/https%3A%2F%2Ffirebasestorage.googleapis.com%2Fv0%2Fb%2Fasl-recognition-d264d.appspot.com%2Fo%2Fasl-test2.mp4%3Falt%3Dmedia%26token%3D0e930fb2-1bc6-4575-9dcf-5152132eb4f4
# W LINKU DO WIDEO NALEŻY ZAMIENIĆ ZNAKI SPECJALNE (NP / = ?) WEDŁUG SCHEMATU ZE STRONY:
# https://www.w3schools.com/tags/ref_urlencode.ASP
#
# HAVE FUN

