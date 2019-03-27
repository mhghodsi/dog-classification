from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('agg')

import torch
import torch.nn as nn

import os
from torchvision import datasets
import torchvision.transforms as transforms
import helper as fn
import numpy as np
from glob import glob
from PIL import Image
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_classes():
    classes = []
    listoffolders = os.listdir('../dogImages/train/')
    listoffolders.sort()
    for f in listoffolders:
        classes.append(f.split(".")[1].replace("_", " "))
    return classes

def VGG16_predict(img_path):
    img = Image.open(img_path)
    transform_pipeline = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
    # pre-process the image
    img = transform_pipeline(img)
    img = img.unsqueeze(0).to(device)
    # change the image tensor to a variable
    img = Variable(img)

    prediction = VGG16(img)
    prediction = prediction.data.cpu().numpy().argmax()
    return prediction # predicted class index

def predict_breed_transfer(img_path, class_names):
    # load the image and return the predicted breed
    img = Image.open(img_path)
    transform_pipeline = transforms.Compose([
                                        transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])
    img = transform_pipeline(img)
    img = img.unsqueeze(0).to(device)
    img = Variable(img)

    model_transfer.load_state_dict(torch.load('models/model_transfer.pt'))
    prediction = model_transfer(img)
    prediction = prediction.data.cpu().numpy().argmax()
    return class_names[prediction] # predicted class name

def run_app(img_path, class_names):
    ## handle cases for a human face, dog, and neither
    print(img_path)
    try:
        if fn.dog_detector(img_path):
            print(img_path)
            print("Who let the dog out! :)")
            # display_img(img_path)
            print("Let me guess your breed ...")
            print("Predicted breed : {}".format(predict_breed_transfer(img_path, class_names)))
        elif fn.face_detector(img_path):
            print("Hello, human!")
            print(img_path)
            # display_img(img_path)
            print("You look like a ...")
            print("{}".format(predict_breed_transfer(img_path, class_names)))
        else:
            print("Ops ... neither a human, nor a dog! What are you then?")
            # display_img(img_path)
            print(img_path)
            print("I can see ...")
            print("{}, is that right? ;)".format(fn.predict_content(img_path)))
    except (IOError, SyntaxError) as e:
        print('ops ... serious error! {}'.format(e))

def main():
    class_names = get_classes()
    sample_photos = np.array(glob("sample_photos/*"))
    for f in sample_photos:
        run_app(f, class_names)

if __name__ == '__main__':
    main()
