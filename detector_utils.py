import cv2
from PIL import Image
import torchvision.transforms as transforms


class FaceDetector:

    def __init__(self):
        # extract pre-trained face detector
        self.face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

    def face_detector(self, image):
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray)
        return len(faces) > 0

    def VGG16_predict(img_path):
        """Use pre-trained VGG-16 model to obtain index corresponding to
        predicted ImageNet class for image at specified path

        Args:
            img_path: path to an image

        Returns:
            Index corresponding to VGG-16 model's prediction
        """

        image = process_image(img_path)

        model = VGG16
        top_k = 1

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