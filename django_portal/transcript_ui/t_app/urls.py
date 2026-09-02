# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("", views.transcript_ui, name="transcript_ui"),
    path("result/<str:result_id>/", views.transcript_result, name="transcript_result"),
]