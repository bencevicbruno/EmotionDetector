#!/usr/bin/env python3

#imports
import cv2
import os
import dlib
import numpy as np

# Setting up detector for face landmarks
frontalface_detector = dlib.get_frontal_face_detector()
landmark_predictor=dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

def distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return (dx*dx + dy*dy)**0.5

def calculate_landmark_distances(landmarks):
    no_landmarks = len(landmarks)
    distances = []
    
    for i in range(no_landmarks):
        for j in range(no_landmarks):
            if i >= j: continue
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

def process_image(path):
    image = cv2.imread(path)
    landmarks = get_landmarks(image)
    distances = calculate_landmark_distances(landmarks)
    
    return distances

def write_distances(distances, emotions):
    tags = []
    
    for i in range(68):
        for j in range(68):
            if i >= j: continue
            firstDot = f"{i}" if i >= 10 else f"0{i}"
            secondDot = f"{j}" if j >= 10 else f"0{j}"
            tags.append(f"d{firstDot}{secondDot}")
    tags.append("emotion")
    
    with open("distances_new.csv", "w") as file:
        column_names = ",".join(tags)
        file.write(column_names)
        file.write("\n")

        for index, d in enumerate(distances):
            line = ",".join(d)
            file.write(line)
            file.write(",")
            file.write(emotions[index])
            file.write("\n")

def process_test_data():
    distances = []
    emotions = []
    
    for (root, _, files) in os.walk('test_data', topdown=True):
        for file in files:
            path = f"{root}/{file}"
            print(path)
            current_distances = process_image(path)
            distances.append(current_distances)

            if "angry" in path: emotions.append("0")
            elif "happy" in path: emotions.append("1")
            elif "neutral" in path: emotions.append("2")
            elif "sad" in path: emotions.append("3")
            elif "surprised" in path: emotions.append("4")
        break

    write_distances(distances, emotions)
    
process_test_data()
