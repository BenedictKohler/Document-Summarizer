# Collaborators: Daniel, Ben, Jon, Josh, Erin
# Description: File used for routing within our forums app
# Date: 12/6/2020

from django.urls import path
from . import views
from .views import PostListView, PostDetailView, PostCreateView, PostUpdateView, PostDeleteView

# Routes for the forum pages
urlpatterns = [
    path('', PostListView.as_view(), name='forum-home'),
    path('post/<int:pk>/', PostDetailView.as_view(),name ='post-detail'),
    path('about/', views.about, name='forum-about'), 
    path('testExternalFunction/',views.testExternalFunction,\
        name='testExternalFunction'),
    path('post/new/', PostCreateView.as_view(),name ='post-create'), 
    path('post/<int:pk>/update', PostUpdateView.as_view(),name ='post-update'),
    path('post/<int:pk>/delete', PostDeleteView.as_view(),name ='post-delete'),
]
 
