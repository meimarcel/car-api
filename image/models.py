from django.db import models

# Create your models here.
class Image(models.Model):
    image = models.CharField(max_length=400)
    timestamp = models.DateTimeField(auto_now=False, auto_now_add=True)
    
    
    def __str__(self):
        return self.image