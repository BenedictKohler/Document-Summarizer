# Collaborators: Daniel, Ben, Jon, Josh, Erin
# Description: URLs used for routing for the summary app
# Date: 12/6/2020

from django.urls import path
from . import views

# Route to the main summarization page

urlpatterns = [
    path('', views.home, name='summarize-home')


]
