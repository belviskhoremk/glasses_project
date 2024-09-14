# user/views.py

from django.shortcuts import render, redirect
from django.urls import reverse_lazy
from django.views import View
from .models import User
from .forms import UserRegistrationForm, UserLoginForm
from django.contrib.auth import authenticate, login

class RegisterView(View):
    template_name = 'registration/register.html'

    def get(self, request):
        form = UserRegistrationForm()
        return render(request, self.template_name, {'form': form})

    def post(self, request):
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print("form is valid")
            user = form.save()
            login(request, user)
            return redirect(reverse_lazy('login'))
        print("form is invalid")# Redirect to the home page or any other page after registration
        return render(request, self.template_name, {'form': form})

class LoginView(View):
    template_name = 'registration/login.html'

    def get(self, request):
        form = UserLoginForm()
        return render(request, self.template_name, {'form': form})

    def post(self, request):
        form = UserLoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect(reverse_lazy('glasses_landing'))  # Redirect to the home page or any other page after login
            else:
                form.add_error(None, 'Invalid username or password')
        return render(request, self.template_name, {'form': form})