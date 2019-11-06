from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import matplotlib.pyplot as plt
import io, glob, cv2
from PIL import Image, ImageDraw
import PIL, PIL.Image
from io import BytesIO
import base64
import random
import pickle as pkl
from bs4 import BeautifulSoup as bsu
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from django.core.mail import send_mail
from django.conf import settings
from keras.models import Sequential
from keras.layers import Dropout, Dense, Conv2D, MaxPooling2D, Flatten
from keras.datasets.mnist import load_data
from keras.utils import to_categorical
from . import predict
import keras 

# load the pre-requried things
global files, eng_char

files = list(glob.glob("static/predict_image/files/*"))
'''
final_pickle = open("static/model/Final_Character_Upper_Case_CNN_64_32_128_model.pkl","rb")
model_char = pkl.load(final_pickle)
final_pickle.close()

final_pickle = open("static/model/Final_Digit_CNN_64_128_32_model.pkl","rb")
model_digi = pkl.load(final_pickle)
final_pickle.close()'''





def index(request):
    if request.method =="GET":
        return render(request,'index.html')
    elif request.method =="POST":
        name = request.POST.get("name")
        phone = request.POST.get("phone")
        email =[request.POST.get("email")]
        print(name,phone,email)
        feedback = request.POST.get("feedback")
        subject = 'Thanks to Contact Us'
        message = f' Greetings : {name} ,\n It is pleasure to hear from you. Our technical team would reach you soon. Happy Security.'
        email_from = settings.EMAIL_HOST_USER
        send_mail(subject,message,email_from,email)
        our_email = ['mail_id@gmail.com']
        send_mail('Someone Contacted',f'Contacted Person ,\n Name : {name}\n Phone : {phone}\n Message is :  {feedback}',email_from,our_email)
    return render(request,'index.html')
    
def fileupload(request):
    return render(request,'./FileUpload/index.html')

def prediction(request):
    if request.method =="GET":
        img_files = files
        values_to_return = {'file_name':img_files}
        return render(request,'./Prediction/index.html',context=values_to_return)
    elif request.method =="POST":
        name = request.POST.get("img_name")
        model_type = request.POST.get("type")
        if model_type == 'char':
            eng_char = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N",
            "O","P","Q","R","S","T","U","V","W","X","Y","Z"]
            keras.backend.clear_session()
            final_pickle = open("static/model/Final_Character_Upper_Case_CNN_64_32_128_model.pkl","rb")
            model_char = pkl.load(final_pickle)
            final_pickle.close()
            file_name = name
            values = predict.runner(file_name,model_char)
            image_predict = values[0]
            y_preds = values[1]
            y_preds = eng_char[y_preds[0]-1]

            image = cv2.imread(file_name)
            is_success, im_buf_arr = cv2.imencode(".png", image)
            byte_im = im_buf_arr.tobytes()
            graphic = base64.b64encode(byte_im)
            image_ = graphic.decode('utf-8')
            values_to_return = {'img_first':image_, 'file_names':file_name, 'predict':y_preds, 'img_second':image_predict}
            return render(request,'./Prediction/index1.html',context=values_to_return)
        elif model_type == 'digi':
            keras.backend.clear_session()
            final_pickle = open("static/model/Final_Digit_CNN_64_128_32_model.pkl","rb")
            model_char = pkl.load(final_pickle)
            final_pickle.close()
            file_name = name
            values = predict.runner(file_name,model_char)
            image_predict = values[0]
            y_preds = values[1]

            image = cv2.imread(file_name)
            is_success, im_buf_arr = cv2.imencode(".png", image)
            byte_im = im_buf_arr.tobytes()
            graphic = base64.b64encode(byte_im)
            image_ = graphic.decode('utf-8')
            values_to_return = {'img_first':image_, 'file_names':file_name, 'predict':y_preds, 'img_second':image_predict}
            return render(request,'./Prediction/index1.html',context=values_to_return)


def gallery(request):
    return render(request,'index.html')

def contact(request):
    return render(request,'index.html')

def form(request):
    return render(request,'form.html')
