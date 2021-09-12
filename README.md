# Speech-Emotion-Recognotion
## Audio based emotion detection with WebApp

**Task 1:**
In this task we were given a training and testing dataset comprising of several mp3 files and had to build a machine learning model to detect the emotion in each file

![Alt text](screenshots/1.png?raw=true "Prevent")

**Approach:**
The first step in any automatic speech recognition system is to extract features. So firstly, the given dataset was preprocessed using the librosa library to extract various features from the given audio files. Several augmentations like spectral contrast, spectral bandwidth etc. were used which were then added as features for the given audio. Since the most of the audio files were of small duration (< 3 sec), We used a sampling rate of 45100 to extract more information from each audio. The picture below shows the function code used for preprocessing each audio file.

**Task 2:**
In this task we had to build a web application using the machine learning models used in task 1.

![Alt text](screenshots/3.png?raw=true "Detect")

**Brief Explanation:**
To build the web application, Django framework was used. For this, we had to first create a model API so that we could integrate the models with the website. Below is a screenshot of how the frontend page looks –
The website covers a brief intro about speech emotion recognition. It has the option to upload the audio file (in .wav format) and, in the backend, the uploaded audio file is firstly stored in a media folder. It is then pre-processed using librosa for feature extraction. After that, these extracted features are used by the trained machine learning models to predict the emotions along with the confidence rating. The models (trained on 10-folds of data) take about 15 secs for prediction (may in other local machines depending on CPU power).
