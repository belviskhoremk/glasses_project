from .models import Glasses
from rest_framework import serializers



class GlassesSerializers(serializers.ModelSerializer):

    class Meta:
        model = Glasses

        fields = [
            "glass_id",
            "glass_model",
            "first_name",
            "last_name",
            "language"
        ]