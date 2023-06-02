from rest_framework import serializers
from .models import SkripsiModel


class SkripsiSerializer(serializers.ModelSerializer):
    image = serializers.ImageField()

    class Meta:
        model = SkripsiModel
        fields = "__all__"
