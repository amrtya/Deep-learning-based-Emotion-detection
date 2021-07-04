from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.detect_image, name='detect_image'),
    path('mplayer', views.mplayer, name='mplayer')
]