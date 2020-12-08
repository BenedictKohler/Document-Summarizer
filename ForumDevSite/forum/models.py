# Collaborators: Daniel, Ben, Jon, Josh, Erin
# Description: Creation of the model for posts that we store in our database
# Date: 12/6/2020

from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
from django.urls import reverse

# Creation of the Posts table for the forum
class Post(models.Model):
    # Fields used in our database model
    title = models.CharField(max_length=100)  
    content = models.TextField()
    date_posted = models.DateTimeField(default=timezone.now)
    author = models.ForeignKey(User,on_delete=models.CASCADE) # delete post if 
                                                              # user is deleted
    def __str__(self):
        return self.title
    
    #Gets the unique primary key assigned to each post
    def get_absolute_url(self): 
        return reverse('post-detail', kwargs={'pk': self.pk})
