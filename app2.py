# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 18:11:16 2022

@author: Swen
"""

import numpy as np
import cv2
import streamlit as st
from tensorflow import keras
from keras.models import model_from_json
from tensorflow.keras.utils import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase,RTCConfiguration, VideoProcessorBase, WebRtcMode
import streamlit as st
import speech_recognition as sr
# from flair.models import TextClassifier
# from flair.data import Sentence
r = sr.Recognizer()
# load model
emotion_dict = {0:'frustration', 1 :'engagement', 2: 'engagement', 3:'confusion', 4: 'excitement'}
# load json and create model
json_file = open(r'emotion_model1.json')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)

# load weights into new model
classifier.load_weights(r"emotion_model1.h5")
sentiment=None
# classifier = TextClassifier.load('en-sentiment')
#load face
try:
    face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")
    
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

all_text = []

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # with sr.Microphone() as source:
        #         audio = r.listen(source)
        #         text = r.recognize_google(audio)
        #         print(text)
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

def main():
    # Face Analysis Application #
    st.title("Multimodal Emotion Detection Application")
    activiteis = ["Home", "Webcam Face Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    # st.sidebar.markdown(
    #     """ Developed by Swen Vincent Fereira    
    #         Email : swenfereira@gmail.com  
    #         """)
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Face Emotion detection application using OpenCV, Custom CNN model and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
    #     st.write("""
    #              Instruction while using this APP.
		 
    # 1. Click on the home button and select Webcam Emotion Detection.
    
    # 2. Click on the Start button to start.
		 
    # 3. Allow the webcam access and WebCam window will open afterwardsn.
    
    # 4. It will load the realtime face emotion detection block with the prediction.
    
    # 5. Click on  Stop  to end.
    #              """)
    elif choice == "Webcam Face Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your face emotion")
        # audio_data = st_audio_recorder()
        
        # audio_bytes = st.audio(format='audio/wav')
        # print(audio_bytes)
        # if st.button('Record'):
        # with sr.Microphone() as source:
        webrtc_streamer(key="example", rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=VideoTransformer)
            # print("Say something!")
            # audio = r.listen(source)

            # text = r.recognize_google(audio)
            # if text!=None:
                # sentence = Sentence(text)
                # classifier.predict(sentence)
                # st.sidebar.write("Sentiment:", sentiment.labels)
            # all_text.append(text)
        

    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    Real time face emotion detection application using OpenCV, Custom Trained CNN model and Streamlit.</h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        html_temp4 = """
                             		<div style="background-color:#98AFC7;padding:10px">
                             		<h4 style="color:white;text-align:center;">This Application is developed by Git Over It using Streamlit Framework, Opencv, Tensorflow and Keras library for demonstration purpose. </h4>
                             		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                             		</div>
                             		<br></br>
                             		<br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)

    else:
        pass


if __name__ == "__main__":
    main()
