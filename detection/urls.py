from django.urls import path
from .views import upload_media, webcam_view, webcam_feed

urlpatterns = [
    path('', upload_media, name='upload_media'),
    path('webcam/', webcam_view, name='webcam'),
    path('webcam-feed/', webcam_feed, name='webcam_feed'),
]
