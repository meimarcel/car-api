from django.contrib import admin

from .models import Image
# Register your models here.

class ImageModelAdmin(admin.ModelAdmin):
    class Meta:
        model = Image
        
admin.site.register(Image, ImageModelAdmin)