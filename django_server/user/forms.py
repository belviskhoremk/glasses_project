# user/forms.py

from django import forms
from .models import User
from django.contrib.auth.forms import UserCreationForm

class UserRegistrationForm(UserCreationForm):
    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'username', 'password1', 'password2']

    first_name = forms.CharField(required=True)
    last_name = forms.CharField(required=True)



class UserLoginForm(forms.Form):
    username = forms.CharField()
    password = forms.CharField(widget=forms.PasswordInput)
