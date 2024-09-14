from django.db import models

# Create your models here.
class Glasses(models.Model):
    CHOICE =    {
        ("english" , "english"),
        ("hindi" , "hindi") ,
        ("french" , "french") ,
        ("spanish"  ,"spanish")
    }

    glass_id = models.CharField(max_length = 50)
    glass_model = models.CharField(max_length = 20)
    first_name = models.CharField(max_length = 50)
    last_name = models.CharField(max_length = 50)
    language = models.CharField(max_length = 10 , choices = CHOICE , default = 'english')

