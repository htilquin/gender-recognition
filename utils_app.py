import dlib
import joblib
import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt

from time import time

def start_timer():
    return time()

def stop_timer(start_time):
    return round(time()-start_time, 2)

def hex2RBG(hex_color):
    return tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

@st.cache(suppress_st_warning=True)
def get_face_detector():
    # HOG based detector from dlib
    return dlib.get_frontal_face_detector()

@st.cache(suppress_st_warning=True)
def get_shape_predictor():
    return dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

@st.cache(suppress_st_warning=True)
def get_face_rec_model():
    return dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

@st.cache(suppress_st_warning=True)
def load_trained_classifier() :
    return joblib.load(f'model_clf_dlib.joblib')

def detect_faces(image, detector) :
    face_rectangles = detector(image, 1)
    return face_rectangles

def get_wider_rectangle(rect):
    width = rect.right() - rect.left()
    height = rect.bottom() - rect.top()
    new_rect = dlib.rectangle(
        max(0, int(rect.left() - .05*width)), 
        max(0, int(rect.top() - 0.2*height)), 
        int(rect.right() + .05*width), 
        int(rect.bottom() + 0.2*height))
    return new_rect

def read_image_from_streamlit_cv2(streamlit_image):
    image = cv2.imdecode(np.frombuffer(streamlit_image.read(), np.uint8), 1)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def get_face_embeddings(img, rect, shape_predictor, face_recognition_model):
    shape = shape_predictor(img, rect)
    face_descriptor = face_recognition_model.compute_face_descriptor(img, shape)
    return face_descriptor

def get_gender_from_features(classifier, face_embeddings):
    prediction = classifier.predict(np.array(face_embeddings).reshape(1, -1))[0]
    return "male" if prediction == -1 else "female"

def add_rectangle_on_image(image, rect, color, weight):
    return cv2.rectangle(image,(rect.left(), rect.top()), (rect.right(), rect.bottom()), color, weight)

def grid_display(images, columns):
    image_number = 1
    for _ in range(int(len(images)/columns)+1) :
        cols = st.columns(columns)
        for column in range(0,columns):
            if image_number > len(images) :
                break
            else :
                cols[column].image(images[image_number-1], use_column_width=True)
                image_number += 1

def display_paritometer(pc_women,color_1, color_2):
    angles = [i for i in range(0, 181, 18)]
    graduations = [i for i in range(0, 101, 10)]
    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(111, polar=True)

    ax.set_thetamin(0)
    ax.set_thetamax(180) #demi-cercle
    ax.set_theta_zero_location("W") #change position du 0
    ax.set_theta_direction(-1) #change la direction de graduation

    ax.set_thetagrids(angles, labels=graduations, fontsize=15)
    ax.set_yticklabels([])
    ax.grid(alpha=0)
    plt.text(0.5*np.pi, 70, "Gender-Parity Meter", ha='center', fontsize=20, fontfamily="serif")

    corrected_angle = max(0.01, min(3.135, pc_women*np.pi))
    ax.plot((0, corrected_angle), (0, 50), color=color_1, lw=2)

    ax2 = plt.subplot(111, projection='polar') 
    tick = [ax2.get_rmax(), ax2.get_rmax() * 0.97] 

    for t in np.deg2rad(np.arange(0, 180, 10)): 
        ax2.plot([t, t], tick, lw=2, color=color_2) 

    return fig

footer="""<style>
a:link , a:visited{
color: red;
background-color: transparent;
text-decoration: none;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: gray;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with 💖 by <a style='display: block; text-align: center;' href="https://htilquin.github.io/" target="_blank">Hélène T.</a></p>
</div>
"""