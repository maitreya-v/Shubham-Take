# Face-Emotion-Recognition-

•	For the past ten years, the Indian education system has been undergoing rapid  changes due to  the expansion of web-based learning services, specifically education platforms.  During a lecture in a physical classroom, the teacher can see the students' faces and gauge their mood, and adjust their lecture accordingly, whether they are going fast or slow.  They can identify students who want additional attention, but they are unable to see all  students or access the mood in digital classrooms due to the usage of a video telephony  software application (ex: Zoom). Students are unable to focus on content due to a lack of supervision as a result of this issue. Physical monitoring is limited on digital platforms, but they do come with the power of data and machines that can work for you. Deep learning algorithms may be used to examine its data, which not only solves the surveillance problem but also eliminates human bias from the system.


•	The aim of the project is to create a Facial Emotion Recognition System (FERS) that can detect students' emotional states in e-learning systems that use video conferencing.  This technology instantly conveys the emotional states of the students to the educator in order to create a more engaged educational environment. Our results supported those of other studies that have shown that in e- learning systems, it is possible to observe the motivation level of both the individual and the virtual classroom.


## Our dataset have 7 types of emotion ranging form 0-6:

•	Anger

•	Disgust

•	Fear

•	Happiness

•	Sad

•	Surprise

•	Neutral

After looking bar plot we can observe that majority of the classes belongs to Happy,Sad and Neutral on the otherside anger, Fear and surprise are average and disgust is very low in number.

## Project Approch:-

### Step 1. Build Model

We have used Five different models as follows:

Model 1- Mobilenet Model

Model 2- Dexpression Model

Model 3- CNN Model

Model 4- Densenet Model

Model 5- Resnet Mode 


### Step 2. Real Time Prediction

And then we perform Real Time Prediction on our best model using webcam on Google colab itself.

  - Run webcam on Google Colab
   
  - Load our best Model
  
  - Real Time prediction
 
### Step 3. Deployment

And lastly we have deployed it on Streamlit Docs .

### ● Deployed App Link : 

 https://share.streamlit.io/swenfereira/face-emotion-recognition-/main/app2.py
 
 
### Conclusion:-

● All the models such as Mobilenet, Dexpression, CNN, Densenet, and ResNet were evaluated.The ResNet model was chosen because it had the highest training accuracy of all the models, and its validation accuracy was nearly 72 percent, which is comparable to CNN models.

● As a result, we save this resnet model and use it to predict facial expressions.

● Since, the emotion counts of disgust and surprise images are less therefore on local webcam it hardly detect those emotions.

● Using streamlit, a front-end model was successfully created and ran on a local webserver.The Streamlit web application has been deployed on Streamlit Docs cloud platform.

● Our github repository contains the code we used to create a web application using Streamlit and deploy it on platforms. It was an amazing and fascinating project. This has taught us a lot.

