from django.contrib import admin
from django.urls import path,include
from rest_framework import routers
from . import views
from django.conf.urls.static import static
from emotion import *

urlpatterns = [
    path('',views.Audio_store),
]