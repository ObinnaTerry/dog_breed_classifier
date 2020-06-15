from PIL import Image
import cv2
import os
import json

import torch.nn as nn
import torch
from torchvision import transforms
import torchvision.models as models


class PredictUtils(Exception):

    def __init__(self):
        """Load model, process and predict image.

        The class provides load_checkpoint method for loading a trained model. process_image method for precessing an
        input image for prediction. imshow method for displaying an image. predict method for making a prediction on an image.
        """

        Exception.__init__(self)

        with open('idx_name_map.json', 'r') as file:
            self.cls_map = json.loads(file.read())

    @staticmethod
    def load_model(file_path='model_transfer.pt'):
        """Load a trained model.

        :param file_path: file path to a checkpoint of the model
        :return: model
        """
        if not os.path.exists(file_path):
            raise Exception('Target model could not be found')

        # checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)
        #
        # model = checkpoint['model']
        # print(model)
        # model.classifier = checkpoint['classifier']
        # print('here')
        # model.load_state_dict(checkpoint['state_dict'])
        #
        # for parameter in model.parameters():
        #     parameter.requires_grad = False

        # model.eval()
        model = models.vgg16(pretrained=True)

        classifier_input = model.classifier[0].in_features

        for param in model.parameters():
            param.requires_grad = False

        model.classifier = nn.Sequential(nn.Linear(classifier_input, 1024),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(1024, 256),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),

                                         nn.Linear(256, 133),
                                         nn.LogSoftmax(dim=1))
        model.load_state_dict(torch.load('model_transfer.pt', map_location=torch.device('cpu')))

        return model

    @staticmethod
    def process_image(image):
        """Processes image to be input into a model for prediction.

        Scales, crops, and normalizes a PIL image for a PyTorch model.
        :param image: file path to the image file to be processed.
        :return: a tensor of the processed image.
        """
        # if not os.path.exists(image):
        #     raise Exception('Target image could not be found')

        image = Image.open(image)

        image_norm = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(244),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

        return image_norm(image)

    def predict(self, image):

        """ Predict the class of an image using input model.

        Takes and input image and produces a top top_k prediction of the image class. if a json file mapping categories
        to names is provided, the output prediction is name dictionary of name: probability pairs,
        else output is a dictionary of category: probability pair.

        prediction can be made on a cpu or gpu. To make predictions on gpu, change mode to gpu.
        :param image: image to be predicted. (type: tensor)
        :param model: model to be used for prediction
        :return: dictionary of result
        """
        model = self.load_model()

        if torch.cuda.is_available():
            device = torch.device("cuda")
            model.to(device)
            image = image.to(device)

        else:
            model.cpu()

        model.eval()

        image = self.process_image(image)
        image = image.unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output).item()

        return self.cls_map[str(prediction)]


class Detector(PredictUtils):

    def __init__(self):
        # extract pre-trained face detector
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        # define VGG16 model
        self.VGG16 = models.vgg16(pretrained=True)

        PredictUtils.__init__(self)

    def face_detector(self, image):
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray)
        return len(faces) > 0

    def VGG16_predict(self, image):
        """Use pre-trained VGG-16 model to obtain index corresponding to
        predicted ImageNet class for image at specified path

        Args:
            image: path to an image

        Returns:
            Index corresponding to VGG-16 model's prediction
        """

        image = self.process_image(image)

        model = self.VGG16

        if torch.cuda.is_available():

            device = torch.device("cuda")
            model.to(device)
            image = image.to(device)

        else:
            model.cpu()
        model.eval()
        image = image.unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output).item()

        return prediction

    def dog_detector(self, img_path):

        result = self.VGG16_predict(img_path)

        return result in range(151, 269)
