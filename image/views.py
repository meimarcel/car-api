import copy
import numpy as np
import supervisely_lib as sly
import cv2
import io
import base64
import os
import json
import base64
import os

from django.shortcuts import render
from rest_framework.parsers import MultiPartParser, FormParser
from PIL import Image
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
from random import randint
from django.conf import settings
from .serializers import ImageSerializer
from django.http import FileResponse
from django.http import HttpResponse
from django.http import JsonResponse
from supervisely_lib.imaging import font as sly_font
from django.core.files.base import ContentFile
from io import BytesIO

# Take in base64 string and return cv image
def stringToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    image = Image.open(io.BytesIO(imgdata))
    return np.array(image)

def RGBToString(image_np,format='jpeg'):
    im = Image.fromarray(image_np.astype("uint8"))
    #im.show()  # uncomment to look at the image
    print("-")
    rawBytes = io.BytesIO()
    print("--")
    im.save(rawBytes, "JPEG")
    print("---")
    rawBytes.seek(0)  # return to the start of the file
    print("----")
    encoded_string = base64.b64encode(rawBytes.read())
    return 'data:image/%s;base64,%s' % (format, encoded_string)

def image_as_base64(image_file, format='jpeg'):
    """
    :param `image_file` for the complete path of image.
    :param `format` is format for image, eg: `png` or `jpg`.
    """
    if not os.path.isfile(image_file):
        return None
    
    encoded_string = ''
    with open(image_file, 'rb') as img_f:
        encoded_string = base64.b64encode(img_f.read())
    return 'data:image/%s;base64,%s' % (format, encoded_string)

def replace_exceto_aspas(s):
    if '"\'"' == s:
        return "'"
    return s.replace("'", "")

token = os.environ.get("S_TOKEN")
SERVER = 'https://app.supervise.ly'

api = sly.Api(SERVER, token)

#sly_font.DEFAULT_FONT_FILE_NAME='LiberationMono-Italic.ttf'

# Create your views here.
img_width = 416
img_height = 416
model_id = 22663
file_types = ['png','jpg','jpeg']   

class ImagePredict(APIView):
    
    serializer_class = ImageSerializer
    
    def post(self, request, format=None):
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            try:
                ## modelo novo
                img_req = request.FILES['imageFile']
                print(request.GET, request.POST)
                print("imagem recebida...")
                
                if(img_req.name.split(".")[-1].lower() not in file_types):
                    return Response({'message': 'Certifique-se que sua imagem está entre os formatos de PNG, JPG ou JPEG'}, status.HTTP_400_BAD_REQUEST)
            
                print("convertendo imagem recebida...")
                imagem_rec = img_req.file.read()

                image = np.array(Image.open(io.BytesIO(imagem_rec)))
                print("imagem convertida...")
                meta_json = api.model.get_output_meta(model_id)
                model_meta = sly.ProjectMeta.from_json(meta_json)
                
                print("classificando imagem...")
                ann_json = api.model.inference(model_id, image)
                ann = sly.Annotation.from_json(ann_json, model_meta)
                
                quantidade = 0
                for objeto in ann_json['objects']:
                    if(objeto['classTitle'] == 'car_model'):
                        quantidade += 1
                
                print("Quantidade de carros: "+str(quantidade))

                canvas_draw_contour = image.copy()
                ann.draw_contour(canvas_draw_contour, thickness=4)

                im = Image.fromarray(canvas_draw_contour)

                #response = HttpResponse(content_type="image/jpeg")
                #im.save(response, "JPEG")
                
                #print("enviando...")
                #return response 
                
                print("Encodando imagem...")
                processed_string = RGBToString(canvas_draw_contour)
                print("Formatando response...")
                
                format, imgstr = processed_string.split(';base64,b')
                a = replace_exceto_aspas(imgstr)
                
                return JsonResponse({"imageFile": a, "quantidade": quantidade})         
                                
            except Exception as message:
                return Response({'message': str(message)}, status.HTTP_400_BAD_REQUEST)
            
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class ImagePredictB64(APIView):
    
    serializer_class = ImageSerializer
    
    def post(self, request, format=None):
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            try:                          
                image_data = request.POST['imageFile']
                
                print("imagem recebida...")
                print("convertendo imagem recebida...")
                
                image = stringToRGB(image_data)
                
                print("Imagem convertida...")
                print("printando modelo*")  
                  
                meta_json = api.model.get_output_meta(model_id)
                print(meta_json)
                model_meta = sly.ProjectMeta.from_json(meta_json)
                print("printando modelo**")    

                print("classificando imagem...")
                ann_json = api.model.inference(model_id, image)
                ann = sly.Annotation.from_json(ann_json, model_meta)

                quantidade = 0
                for objeto in ann_json['objects']:
                    if(objeto['classTitle'] == 'car_model'):
                        quantidade += 1
                        
                print("Quantidade de carros: "+str(quantidade))
                
                canvas_draw_contour = image.copy()
                ann.draw_contour(canvas_draw_contour, thickness=4)

                im = Image.fromarray(canvas_draw_contour)

                print("Encodando imagem...")
                processed_string = RGBToString(canvas_draw_contour)
                print("Formatando response...")
                
                format, imgstr = processed_string.split(';base64,b')
                a = replace_exceto_aspas(imgstr)
                
                return JsonResponse({"imageFile": a, "quantidade": quantidade})         
            except Exception as message:
                return Response({'message': str(message)}, status.HTTP_400_BAD_REQUEST)
            
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)