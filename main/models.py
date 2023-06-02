from django.db import models

# Create your models here.


class SkripsiModel(models.Model):
    image = models.ImageField(upload_to="images/")
