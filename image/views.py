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
from PIL import Image, GifImagePlugin

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

def analyseImage(img):
    '''
    Pre-process pass over the image to determine the mode (full or additive).
    Necessary as assessing single frames isn't reliable. Need to know the mode 
    before processing all frames.
    '''
    results = {
        'size': img.size,
        'mode': 'full',
    }
    try:
        while True:
            if img.tile:
                tile = img.tile[0]
                update_region = tile[1]
                update_region_dimensions = update_region[2:]
                if update_region_dimensions != img.size:
                    results['mode'] = 'partial'
                    break
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    return results

token = os.environ.get("S_TOKEN")
SERVER = 'https://app.supervise.ly'

api = sly.Api(SERVER, token)

#sly_font.DEFAULT_FONT_FILE_NAME='LiberationMono-Italic.ttf'

# Create your views here.
img_width = 416
img_height = 416
model_id = 22663
file_types = ['png','jpg','jpeg','gif']   

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
    
class GifPredict(APIView):
    
    serializer_class = ImageSerializer
    def post(self, request, format=None):
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            try:
                ## modelo novo
                img_req = request.FILES['imageFile']
                print(request.GET, request.POST)
                print("gif recebido...")
                
                if(img_req.name.split(".")[-1].lower() != "gif"):
                    return Response({'message': 'Certifique-se que você está enviando um GIF'}, status.HTTP_400_BAD_REQUEST)
            
                print("convertendo gif recebido...")
                imagem_rec = img_req.file.read()

                gif = Image.open(io.BytesIO(imagem_rec))
                images = []
                '''
                Iterate the GIF, extracting each frame.
                '''
                mode = analyseImage(gif)['mode']

                p = gif.getpalette()
                last_frame = gif.convert('RGBA')
                gif.seek(0)

                try:
                    while True:
                        '''
                        If the GIF uses local colour tables, each frame will have its own palette.
                        If not, we need to apply the global palette to the new frame.
                        '''
                        if not gif.getpalette():
                            gif.putpalette(p)
                        
                        new_frame = Image.new('RGBA', gif.size)
                        
                        '''
                        Is this file a "partial"-mode GIF where frames update a region of a different size to the entire image?
                        If so, we need to construct the new frame by pasting it on top of the preceding frames.
                        '''
                        if mode == 'partial':
                            new_frame.paste(last_frame)
                        
                        new_frame.paste(gif, (0,0), gif.convert('RGBA'))
                        images.append(np.array(new_frame.convert("RGB")))

                        last_frame = new_frame
                        gif.seek(gif.tell() + 1)
                except EOFError:
                    pass
                print(len(images))
                print("gif convertido...")
                meta_json = api.model.get_output_meta(model_id)
                model_meta = sly.ProjectMeta.from_json(meta_json)
                
                print("classificando gif...")
                responseImages = []
                quantidades = []
                for img in images:
                    ann_json = api.model.inference(model_id, img)
                    ann = sly.Annotation.from_json(ann_json, model_meta)
                    
                    quantidade = 0
                    for objeto in ann_json['objects']:
                        if(objeto['classTitle'] == 'car_model'):
                            quantidade += 1
                    quantidades.append(quantidade)
                    
                    print("Quantidade de carros: "+str(quantidade))

                    canvas_draw_contour = img.copy()
                    ann.draw_contour(canvas_draw_contour, thickness=4)

                    im = Image.fromarray(canvas_draw_contour)
                
                    print("Encodando imagem...")
                    processed_string = RGBToString(canvas_draw_contour)
                    print("Formatando response...")
                
                    format, imgstr = processed_string.split(';base64,b')
                    responseImages.append(replace_exceto_aspas(imgstr))
                
                return JsonResponse({"imageFile": responseImages, "quantidade": quantidades})         
                                
            except Exception as message:
                return Response({'message': str(message)}, status.HTTP_400_BAD_REQUEST)
            
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)