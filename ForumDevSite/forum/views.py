# Collaborators: Daniel, Ben, Jon, Josh, Erin
# Description: Views file used to handle data and requests for each page on our forums app
# Date: 12/6/2020

from django.shortcuts import render
from django.http import HttpResponse
from .models import Post
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from .testClass import djangoMethodTest
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin

# These are the views for the home and forum pages

#Sends posts objects to home page for rendering
def home(request):
    context = {'posts': Post.objects.all()}
    return render(request, 'forum/home.html', context)

#Sends posts objects to home page for rendering 
class PostListView(ListView):
    model = Post
    template_name = 'forum/home.html'
    context_object_name = 'posts'
    ordering = ['-date_posted']

#Displays details for individual posts when individual posts are opened
class PostDetailView(DetailView):
    model = Post

def testExternalFunction(request):
    return render(request, 'forum/externalTest.html', \
        {'testPrint':djangoMethodTest('forum/testLoadData.txt')})

def about(request):
    return render(request, 'forum/about.html', {'title':'About'})

#Used to create new posts
class PostCreateView(LoginRequiredMixin, CreateView):
    model = Post
    fields = ['title', 'content']
    #Checks if the post created is valid
    def form_valid(self,form):
        form.instance.author = self.request.user
        return super().form_valid(form)

#Used to update posts
class PostUpdateView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Post
    fields = ['title', 'content']

    def form_valid(self,form):
        form.instance.author = self.request.user
        return super().form_valid(form)

    #Makes sure the author updating the post is the author who creatde the post
    def test_func(self):
        post = self.get_object()
        if self.request.user == post.author:
            return True
        return False

#Used to delete posts
class PostDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Post
    success_url = '/forum'

    #Makes sure the author deleting the post is the author who creatde the post
    def test_func(self):
        post = self.get_object()
        if self.request.user == post.author:
            return True
        return False

