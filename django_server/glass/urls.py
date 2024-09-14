from django.urls import path
from . import views

urlpatterns = [
    path('', views.glasses_landing, name='glasses_landing'),
    path('register/', views.register_glasses, name='register_glasses'),
    path('glass_detail/' ,views.glass_detail , name = 'glass_detail')

]