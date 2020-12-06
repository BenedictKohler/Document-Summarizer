# Collaborators: Daniel, Ben, Jon, Josh, Erin
# Description: Overrides the exisitng UserCreationForm with our created UserRegisterForm to add email into the model fields
# Date: 12/6/2020

from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm

# This is a generic django form used for login and registration

class UserRegisterForm(UserCreationForm):
    email = forms.EmailField()

    class Meta:
        model = User
        fields =  ['username', 'email', 'password1', 'password2']
