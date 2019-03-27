# helper functions
import json
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import torch
import torchvision.models as models

# define VGG16 model
VGG16 = models.vgg16(pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_content(img_path):
    with open('imagenet1000_labels.json') as f:
        data = json.load(f)
    labels = {int(key): value for key, value in data.items()}
    prediction = VGG16_predict(img_path)
    return labels[prediction]

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = VGG16_predict(img_path)
    return True if prediction >= 151 and prediction <= 268 else False # true/false
