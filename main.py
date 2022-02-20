#!/usr/bin/env python3

#imports
import cv2
import os
import dlib
import numpy as np
import urllib.request
import json
from subprocess import Popen, PIPE
import subprocess

# Setting up detector for face landmarks
frontalface_detector = dlib.get_frontal_face_detector()
landmark_predictor=dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

# Setting up the Window
WINDOW = "Emotion Detector"
cam = cv2.VideoCapture(0)
cv2.namedWindow(WINDOW)

def distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return (dx*dx + dy*dy)**0.5

def calculate_landmark_distances(landmarks):
    no_landmarks = len(landmarks)
    distances = []
    
    for i in range(no_landmarks):
        for j in range(no_landmarks):
            if i == j: continue
            distances.append(distance(landmarks[i], landmarks[j]))

    max_distance = max(distances)

    for i in range(len(distances)):
        distances[i] /= max_distance
        distances[i] = str(distances[i])

    return distances       

def get_landmarks(image):
    faces = frontalface_detector(image, 1)

    if len(faces):
        landmarks = [(p.x, p.y) for p in landmark_predictor(image, faces[0]).parts()]
        return landmarks
    else:
        return None

def save_distances(distances):
    line = ','.join(distances)
    line = line + ', 0'

    with open('distances.csv', 'w') as file:
        file.write(line)

def paint_landmarks(image, landmarks):
    radius = -1
    thickness = 2
    image_copy = image.copy()
    for (x, y) in landmarks:
        cv2.circle(image_copy, (x, y), thickness, (255, 0, 0), radius)
    return image_copy

def fetch_emotion():
    process = subprocess.run('./RUAP/bin/Debug/RUAP.exe')
    code = process.returncode -10
    text = "unknown"
    
    if code == 0: text = "Angry"
    elif code == 1: text = "Happy"
    elif code == 2: text = "Sad"
    elif code == 3: text = "Neutral"
    elif code == 4: text = "Surprised"

    return text

def show_emotion(emotion, cam_image):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (0, 100)
    fontScale = 2
    fontColor = (0, 0, 0)
    thickness = 3
    lineType = 2
    
    cv2.putText(cam_image, emotion,
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType)
    cv2.imshow(WINDOW, cam_image)


ret, frame = cam.read()
if ret:
    cv2.imshow(WINDOW, frame)

hold_frame = False 
while True:
    ret, frame = cam.read()
    if not ret: break

    if not hold_frame:
        cv2.imshow(WINDOW, frame)

    key = cv2.waitKey(1)
    if key % 256 == 27: break # ESC pressed
    elif key % 256 == ord('b'):
        if not hold_frame:
            hold_frame = True
            landmarks = get_landmarks(frame)
            image = paint_landmarks(frame, landmarks)
            distances = calculate_landmark_distances(landmarks)
            save_distances(distances)

            emotion = fetch_emotion()
            show_emotion(emotion, image)
        else:
            hold_frame = False

cam.release()
cv2.destroyAllWindows()
