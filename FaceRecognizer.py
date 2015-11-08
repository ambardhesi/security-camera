import re
import os
import cv2
import numpy as np

# FaceRecognizer is a class which takes a training path
# function get_data(path) obtains the data from the path
# function train_data() trains the recognizer
# function predict_data(image_path) takes an image and predicts who the person is

class FaceRecognizer:
    def __init__(self, train_data_path = "/home/pi/OpenCVStuff/SecurityCamera/KnownPeople"):
        self.train_data_path = train_data_path
        self.recognizer = cv2.face.createLBPHFaceRecognizer()
        self.faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
 
    def get_data(self):
        """Takes all the data from path stored in self.train_data_path, and creates a data set.
        Returns two arrays :
        images : All the training images stored as numpy arrays.
        labels : The corresponding labels of the images. 
        Eg of data location ~/home/pi/SecurityCamera/KnownPeople/s1/1.jpg
        Where s1, s2, s3... are the different people in the database,
        and 1.jpg, 2.jpg... are the different images for each person."""

        path = self.train_data_path
        images = []
        labels = []
    
        # loop over all folders and files in the path
        for dirpath, dirnames, filenames in os.walk(path):
            for name in filenames:
                image_path = os.path.join(dirpath, name)    
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = np.array(image, 'uint8')
                faces = self.faceCascade.detectMultiScale(image) # detect for face in the image
                               
                # append the detected face to images, and label to labels
                for (x, y, w, h) in faces:
                    images.append(image[y: y + h, x: x + w])
                    label_container = os.path.split(image_path)[0].split('/')[6]
                    label_regex = re.compile(r"\D(\d(\d)*)?")
                    label_search = label_regex.search(label_container)
                    label = int(label_search.group(1))
                    labels.append(label)
                    #cv2.imshow("Adding face to training set...", image[y: y + h, x: x + w])
                    #cv2.waitKey(5)
        labels = np.asarray(labels)
        return images, labels        
    
    def train_data(self):
        """Trains the Local Binary Patterns Histogram recognizer from data taken from get_data()"""
        
        images, labels = self.get_data()
        self.recognizer.train(images, labels)

       
    def predict(self, image_path):
        """Predicts the correct person using the LBPH Recognizer.
        image_path : string which contains the path of person who is to be recognized.
        prediction : integer which is the prediction of the person (1, 2, 3...)
        conf : float which is the confidence level of the prediction (lower is better)"""
       
        predict_image = cv2.imread(image_path)
        predict_image = cv2.cvtColor(predict_image, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(predict_image)
        
        for (x, y, w, h) in faces:
            prediction, conf = self.recognizer.predict(predict_image[y: y + h, x: x + w])
        return (prediction, conf)
        
       

    



