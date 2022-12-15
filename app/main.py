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
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(
    8, 8))  # applying CLAHE to L-channel   - feel free to try different values for the limit and grid size
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))  # merge the CLAHE enhanced L-channel with the a and b channel
    image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)  # onverting image from LAB Color model to BGR color spcae

    # detections
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # color conversion BGR 2 RGB
    image.flags.writeable = False  # image is no longer writeable
    results = model.process(image)  # make prediction
    image.flags.writeable = True  # image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # color conversion RGB 2 BGR
    return image, results


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

        if (results.pose_landmarks.landmark[11].visibility > 0.85 and results.pose_landmarks.landmark[
            12].visibility > 0.85):  # middle point between choulders
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
            if (results.pose_landmarks.landmark[i].visibility > 0.85):
                if (i % 2 == 0):
                    pose_right.append(results.pose_landmarks.landmark[i].x - face[0])
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


# FUNCTIONS FOR EXTRACTING SEQUENCE

def init_global_variables_on_zero(all_variables=True):
    global frame_num, all_keypoints
    global number_of_blank_pose_left, number_of_blank_pose_right, number_of_blank_left_hand, number_of_blank_right_hand
    global vector_of_blank_pose_left, vector_of_blank_pose_right, vector_of_blank_left_hand, vector_of_blank_right_hand
    global all_frames_pose_left, all_frames_pose_right, all_frames_left_hand, all_frames_right_hand

    number_of_blank_pose_left = 0  # -- bez tego bd dzialac ale mozecie zakodzic na poziej bo bd potrzebne
    number_of_blank_pose_right = 0
    number_of_blank_left_hand = 0
    number_of_blank_right_hand = 0

    vector_of_blank_pose_left = []
    vector_of_blank_pose_right = []
    vector_of_blank_left_hand = []
    vector_of_blank_right_hand = []

    if all_variables:  # -- tylko to wam na razie potrzebne
        all_frames_pose_left = []
        all_frames_pose_right = []
        all_frames_left_hand = []
        all_frames_right_hand = []

        frame_num = 1
        all_keypoints = []


def counting_number_of_blanks():
    global frame_num, all_keypoints
    global number_of_blank_pose_left, number_of_blank_pose_right, number_of_blank_left_hand, number_of_blank_right_hand
    global vector_of_blank_pose_left, vector_of_blank_pose_right, vector_of_blank_left_hand, vector_of_blank_right_hand
    global all_frames_pose_left, all_frames_pose_right, all_frames_left_hand, all_frames_right_hand
    global pose_left, pose_right, left_hand, right_hand

    # checking existence of landmarks on body and hands
    if len(left_hand) == 0:
        left_hand = np.zeros(
            42)  # -- tylko to wam na razie potrzebne, bez reszty bd dzialac ale mozecie zakodzic na poziej bo bd potrzebne
        vector_of_blank_left_hand.append(1)
    else:
        vector_of_blank_left_hand.append(0)

    if len(right_hand) == 0:
        right_hand = np.zeros(42)  # -- tylko to wam na razie potrzebne, same here
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

    # number of blanks in last 13 frames not including last one    #  ---  bez tego tez bd dzialac ale mozecie zakodzic na poziej bo bd potrzebne
    if (frame_num > sequence_length):
        vector_of_blank_pose_left = vector_of_blank_pose_left[-13:-1]
        vector_of_blank_pose_right = vector_of_blank_pose_right[-13:-1]
        vector_of_blank_left_hand = vector_of_blank_left_hand[-13:-1]
        vector_of_blank_right_hand = vector_of_blank_right_hand[-13:-1]

    number_of_blank_pose_left = np.count_nonzero(vector_of_blank_pose_left)
    number_of_blank_pose_right = np.count_nonzero(vector_of_blank_pose_right)
    number_of_blank_left_hand = np.count_nonzero(vector_of_blank_left_hand)
    number_of_blank_right_hand = np.count_nonzero(vector_of_blank_right_hand)


# Updating hole story for single part
def update_all_frames():
    global frame_num, all_keypoints
    global number_of_blank_pose_left, number_of_blank_pose_right, number_of_blank_left_hand, number_of_blank_right_hand
    global vector_of_blank_pose_left, vector_of_blank_pose_right, vector_of_blank_left_hand, vector_of_blank_right_hand
    global all_frames_pose_left, all_frames_pose_right, all_frames_left_hand, all_frames_right_hand
    global pose_left, pose_right, left_hand, right_hand

    all_frames_pose_left.append(pose_left)
    all_frames_pose_right.append(pose_right)
    all_frames_left_hand.append(left_hand)
    all_frames_right_hand.append(right_hand)

    # extracting last 12 frames    #- orginalnie bedzie nieco inaczej # extracting last 13 frames i  [-13:] ale to na pozniej
    if (True):  # tymczasowo dla tej wersji kodu
        all_frames_pose_left = all_frames_pose_left[-12:]
        all_frames_pose_right = all_frames_pose_right[-12:]
        all_frames_left_hand = all_frames_left_hand[-12:]
        all_frames_right_hand = all_frames_right_hand[-12:]


# EXTRACTING SEQUENCE OF FRAMES FOR PREDICTION

def extract_sequence(results):
    global frame_num, all_keypoints
    global number_of_blank_pose_left, number_of_blank_pose_right, number_of_blank_left_hand, number_of_blank_right_hand
    global vector_of_blank_pose_left, vector_of_blank_pose_right, vector_of_blank_left_hand, vector_of_blank_right_hand
    global all_frames_pose_left, all_frames_pose_right, all_frames_left_hand, all_frames_right_hand
    global pose_left, pose_right, left_hand, right_hand

    # extracting keypoints
    face, pose_left, pose_right, left_hand, right_hand = extract_keypoints(results)

    # if there is no landmarks
    if len(face) == 0:
        print("nic nie ma na ekraie")
        init_global_variables_on_zero()
        return 404

    # preapering for prediction
    else:
        # checking how many blanks is in every single part of body in last 12 frames
        ## tutaj jest to wazne o tyle ze jak nie ma detekcji to tworzy wektor 0, blanki was jeszcze nie musza obchodzic
        counting_number_of_blanks()

        # updating whole story for single part
        update_all_frames()

        if (frame_num < 12):
            frame_num += 1
            return 404

        elif (frame_num == 12):
            # collecting sequence of 12 frames (one frame from one side pose and hand)
            for i in range(sequence_length):  # -- to sie zmieni ale dla testu najlatwiej to tak zapisac
                keypoints = np.concatenate([all_frames_pose_right[i], all_frames_right_hand[i]])
                all_keypoints.append(keypoints)
            return 1

# folders info
no_sequences = 50
sequence_length = 12
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['pierdol_sie', 'hello','I_me','need','thanks', 'drink', 'beer'])
colors = [(255, 153, 51) ,(245,117,16), (117,245,16), (16,117,245),(245,117,16), (117,245,16), (16,117,245)]

# importing model
model = keras.models.load_model("app/action2d_e25-_96.h5")

def processVideo(video):
    init_global_variables_on_zero()
    global frame_num, all_keypoints
    global number_of_blank_pose_left, number_of_blank_pose_right, number_of_blank_left_hand, number_of_blank_right_hand
    global vector_of_blank_pose_left, vector_of_blank_pose_right, vector_of_blank_left_hand, vector_of_blank_right_hand
    global all_frames_pose_left, all_frames_pose_right, all_frames_left_hand, all_frames_right_hand
    global pose_left, pose_right, left_hand, right_hand
    print(1)
    sentence = []
    threshold = 0.9

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
            ## tutaj dla testu tylko detekcja dla jednej prawej reki
            if network_choose == 1:
                # making prediction
                res = model.predict(np.expand_dims(all_keypoints, axis=0))[0]
                # collecting 5 last words for top bar
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])
                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # cleaing  -- to sie zmieni ale dla testu najlatwiej to tak zapisac
                all_keypoints = []

        cap.release()
    return {"message": " ".join(sentence)}


@app.get("/")
async def root():
    return {"message": "Online!"}


@app.get("/{full_path:path}")
def predictASL(full_path: str):
    res = processVideo(full_path)

    return res

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

