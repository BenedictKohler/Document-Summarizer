from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from .forms import UserRegisterForm
from django.contrib.auth.decorators import login_required
from forum.models import Post
from summarize.models import Summary

# Handle a user registering
def register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid(): # Make sure passord and username are secure enough and valid
            form.save() # Store the users details
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created! You may login')
            return redirect('forum-home') 
    else:
        form = UserRegisterForm()
    return render(request, 'users/register.html', {'form': form})

# Handle a users saved summaries and posts
@login_required
def profile(request):
    
    if request.method == 'POST' :
        try :
            summary = Summary.objects.get(pk=str(request.POST['action'])) # Delete the selected summary
            summary.delete()
        except :
            pass

    return render(request, 'users/profile.html', {'posts': Post.objects.all(), 'summaries': Summary.objects.all()})
