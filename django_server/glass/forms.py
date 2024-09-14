from django import forms
from .models import Glasses

class GlassesForm(forms.ModelForm):
    class Meta:
        model = Glasses
        fields = ['glass_id', 'glass_model', 'first_name', 'last_name', 'language']
        widgets = {
            'glass_id': forms.TextInput(attrs={'class': 'form-input'}),
            'glass_model': forms.TextInput(attrs={'class': 'form-input'}),
            'first_name': forms.TextInput(attrs={'class': 'form-input'}),
            'last_name': forms.TextInput(attrs={'class': 'form-input'}),
            'language': forms.Select(attrs={'class': 'form-select'}),
        }