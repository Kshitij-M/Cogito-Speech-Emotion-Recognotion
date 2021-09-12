from django.http.response import HttpResponse
from django.shortcuts import render
from rest_framework import viewsets,status
from rest_framework.decorators import api_view
from django.core import serializers
from rest_framework.response import Response
from django.http import JsonResponse,HttpResponseRedirect
from django.contrib import messages
from rest_framework.parsers import JSONParser
from . models import Audio_store
from . forms import AudioForm
import operator
import numpy as np
import pandas as pd
from scipy.stats import skew
import librosa
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')
SAMPLE_RATE = 45100


def Audio_store(request):
    if(request.method == 'POST'):
        form = AudioForm(request.POST,request.FILES or None)
        print(request.FILES)
        if(form.is_valid()):
            form.save()
            ans = get_emotion(f"C:\\Users\\9818k\\Downloads\\CogitoProject\\WebApp\\media\\audio\\{form.cleaned_data['record'].name}")
            for k,v in sorted(ans.items(), key = lambda x: x[1], reverse = True):
                messages.success(request,f"Emotion : {k} and Confidence : {round(v, 2)} %")
    else:        
        form = AudioForm()
    return render(request,'form.html', {'form' : form})

def get_mfcc(path):
    b, _ = librosa.core.load(path, sr = SAMPLE_RATE)
    assert _ == SAMPLE_RATE
    try:
        ft1 = librosa.feature.mfcc(b, sr = SAMPLE_RATE, n_mfcc=20)
        ft2 = librosa.feature.zero_crossing_rate(b)[0]
        ft3 = librosa.feature.spectral_rolloff(b)[0]
        ft4 = librosa.feature.spectral_centroid(b)[0]
        ft5 = librosa.feature.spectral_contrast(b)[0]
        ft6 = librosa.feature.spectral_bandwidth(b)[0]
        ft7 = librosa.feature.spectral_flatness(b)[0]
        ft8 = librosa.feature.melspectrogram(b)[0]
        ft1_trunc = np.hstack((np.mean(ft1, axis = 1), np.std(ft1, axis = 1), skew(ft1, axis = 1), np.max(ft1, axis = 1), np.min(ft1, axis = 1), np.sum(ft1, axis = 1)))
        ft2_trunc = np.hstack((np.mean(ft2), np.std(ft2), skew(ft2), np.max(ft2), np.min(ft2), np.sum(ft2)))
        ft3_trunc = np.hstack((np.mean(ft3), np.std(ft3), skew(ft3), np.max(ft3), np.min(ft3), np.sum(ft3)))
        ft4_trunc = np.hstack((np.mean(ft4), np.std(ft4), skew(ft4), np.max(ft4), np.min(ft4), np.sum(ft4)))
        ft5_trunc = np.hstack((np.mean(ft5), np.std(ft5), skew(ft5), np.max(ft5), np.min(ft5), np.sum(ft5)))
        ft6_trunc = np.hstack((np.mean(ft6), np.std(ft6), skew(ft6), np.max(ft6), np.min(ft6), np.sum(ft6)))
        ft7_trunc = np.hstack((np.mean(ft7), np.std(ft7), skew(ft7), np.max(ft7), np.min(ft7), np.sum(ft7)))
        ft8_trunc = np.hstack((np.mean(ft8), np.std(ft8), skew(ft8), np.max(ft8), np.min(ft8), np.sum(ft8)))
        
        return pd.Series(np.hstack((ft1_trunc, ft2_trunc, ft3_trunc, ft4_trunc, ft5_trunc, ft6_trunc, ft7_trunc, ft8_trunc)))
    except:
        print('bad file')
        return pd.Series([0]*162)

def get_emotion(path):
    arr = get_mfcc(path)
    classes = {0: 'anger',
               1: 'disgust',
               2: 'fear',
               3: 'joy',
               4: 'neutral',
               5: 'sadness',
               6: 'surprise'}
    final_preds = np.zeros((1, 7))
    for fold in range(10):
        model = joblib.load(f"C:\\Users\\9818k\\Downloads\\CogitoProject\\WebApp\\model_weights\\vc_fold_{fold}.pkl")
        final_preds += model.predict_proba(arr.values.reshape(1,-1))

    final_preds /= 10

    postprocess = True
    if postprocess:
        for i in range(len(final_preds)):
            temp = np.argmax(final_preds[i])
            if(temp == 4 and final_preds[i][temp] > 0.5 and final_preds[i][temp] <= 0.51):
                final_preds[i][temp] = 0
    ans = {}
    final_preds = list(final_preds[0])
    for i in range(7):
        ans[classes[i]] = (final_preds[i] * 100)
    return ans