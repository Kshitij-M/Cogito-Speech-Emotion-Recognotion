# Speech-Emotion-Recognotion
## Audio based emotion detection with WebApp

## Task 1:
In this task we were given a training and testing dataset comprising of several mp3 files and had to build a machine learning model to detect the emotion in each file

![Alt text](screenshots/1.png?raw=true "task1")

**Approach:**
The first step in any automatic speech recognition system is to extract features. So firstly, the given dataset was preprocessed using the librosa library to extract various features from the given audio files. Several augmentations like spectral contrast, spectral bandwidth etc. were used which were then added as features for the given audio. Since the most of the audio files were of small duration (< 3 sec), We used a sampling rate of 45100 to extract more information from each audio. The picture below shows the function code used for preprocessing each audio file.

## Task 2:
In this task we had to build a web application using the machine learning models used in task 1.

![Alt text](screenshots/2.jpg?raw=true "task2")

**Brief Explanation:**
To build the web application, Django framework was used. For this, we had to first create a model API so that we could integrate the models with the website. Below is a screenshot of how the frontend page looks –
The website covers a brief intro about speech emotion recognition. It has the option to upload the audio file (in .wav format) and, in the backend, the uploaded audio file is firstly stored in a media folder. It is then pre-processed using librosa for feature extraction. After that, these extracted features are used by the trained machine learning models to predict the emotions along with the confidence rating. The models (trained on 10-folds of data) take about 15 secs for prediction (may vary in other local machines depending on CPU power).

## Running this project
To get this project up and running you should start by having Python installed on your computer. It's advised you create a virtual environment to store your projects dependencies separately. 
Then you can project dependencies in the virtual environment with 
```
pip install -r requirements.txt
```
Then you can change the base paths for model weights and audio file in features/views.py.
Now you can run the project with this command.
```
python manage.py runserver
```
