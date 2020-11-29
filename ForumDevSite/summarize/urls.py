from django.urls import path
from . import views

# Route to the main summarization page

urlpatterns = [
    path('', views.home, name='summarize-home')


]
