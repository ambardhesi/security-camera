from dropbox.client import DropboxOAuth2FlowNoRedirect
from dropbox.client import DropboxClient
from picamera.array import PiRGBArray
from FaceRecognizer import FaceRecognizer
import argparse
import json
import cv2
import numpy as np
import picamera
import time
import datetime
import os
import os.path


# Dictionary of known people. Add more people here if required.
known_people = { 1 : "Ambar"}

# create instance of FaceRecognizer class to get training data, train data, and predict person
FaceRecognizer = FaceRecognizer()

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# set up command line argument parser for JSON file path
argParser = argparse.ArgumentParser()
argParser.add_argument("-c", "--conf", required = True, help = "Path to the JSON config file")
args = vars(argParser.parse_args())

conf = json.load(open(args["conf"]))
dropboxClient = None

if conf["use_dropbox"]:
    # connecting to dropbox
    flow = DropboxOAuth2FlowNoRedirect(conf["dropbox_key"], conf["dropbox_secret"])
    print "[INFO] Authorize this application: {}".format(flow.start())
    authCode = raw_input("Enter auth code here: ").strip()
    
    # grab the dropbox client
    (accessToken, userID) = flow.finish(authCode)
    client = DropboxClient(accessToken)
    print "Account linked!"

# loop over all frames
with picamera.PiCamera() as camera:
	camera.resolution = tuple(conf["resolution"])
	camera.framerate = conf["fps"]
        # Initialise avg frame, which will be used to compare all other frames
        avg = None
        lastUploaded = datetime.datetime.now()
        motionCounter = 0

        rawCapture = PiRGBArray(camera, size = tuple(conf["resolution"]))
        time.sleep(conf["camera_warmup_time"])
        
        for frame in camera.capture_continuous(rawCapture, format = "bgr", use_video_port = True):
            image = frame.array # Store the image as numpy array
            timestamp = datetime.datetime.now() # Store the time at which current frame is processed
            text = "Empty Room" # Default room status

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            faces = faceCascade.detectMultiScale(
	        gray,
		scaleFactor = 1.1,
		minNeighbors = 5,
		minSize = (30, 30),
		flags = cv2.CASCADE_SCALE_IMAGE
		)
            # if 1 or more faces were found, set Room Status to be "Occupied"
            if len(faces):
                text = "Occupied"

            # Draw rectangle around face
	    for (x, y, w, h) in faces:
        	cv2.rectangle(image, (x,y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, "Found : {} face(s)".format(len(faces)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # If avg is None, initialise it
            if avg is None:
                print "Starting model..."
                avg = gray.copy().astype("float")
                rawCapture.truncate(0)
                continue

            # Calculate weighted mean of previous frams and current frame
            # and then subtract weighted mean from current frame
            cv2.accumulateWeighted(gray,avg, 0.5)
            frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
         
            # threshold delta image and dilate to fill in holes
            thresh = cv2.threshold(frameDelta, 5, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations = 2)
            
            # find the contours
            (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # loop over all the contours and if the area of contour is too small, ignore it
            for contour in cnts:
                if cv2.contourArea(contour) < 5000:
                    continue

                # draw the bounding box for contour and update the text
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                text = "Occupied"

            # draw the text and timestamp on the frame
            ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
            cv2.putText(image, "Room Status : {}".format(text), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(image, ts, (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
           
            # Uncomment this line if the live feed is to be seen
            #cv2.imshow("Security Camera", image)

            if text == "Occupied":
                if (timestamp - lastUploaded).seconds >= conf["min_upload_seconds"]:
                    motionCounter += 1
                    if motionCounter >= conf["min_motion_frames"]:
                        if len(faces): # checking if number of faces found is > 0 
                            cv2.imwrite("Captured_Img.jpg", image)
                            image_path = "Captured_Img.jpg"
                            FaceRecognizer.train_data()
                            prediction, confidence = FaceRecognizer.predict(image_path)
                            print prediction, confidence
                            # now check if person belongs to known set of people or not by thresholding confidence level
                            if confidence < 50:
                                print " WELCOME :", known_people[prediction]
                                cv2.putText(image, "Verified User : {}".format(known_people[prediction]), (10, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                cv2.imwrite("Captured_Img.jpg", image)
                            else:
                                print "ALERT! NOT VERIFIED! CALL THE POLICE!"
                                cv2.putText(image, "ALERT! NOT VERIFIED! CALL THE POLICE!", (10, image.shape[0] -30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                cv2.imwrite("Captured_Img.jpg", image)

                        if conf["use_dropbox"]:
                            # check if image file was already created due to presence of faces
                            # else create image file due to motion detection only
                            if not(os.path.isfile("Captured_Img.jpg")):
                                cv2.imwrite("Captured_Img.jpg", image)
                            image_path = "Captured_Img.jpg"
                            path = "{base_path}/{timestamp}.jpg".format(base_path = conf["dropbox_base_path"], timestamp = ts)
                            client.put_file(path, open("Captured_Img.jpg", "rb"))
                            os.remove(image_path)

                        lastUploaded = timestamp
                        motionCounter = 0
            else:
                motionCounter = 0

            cv2.imshow("Security Camera", image)

            rawCapture.truncate(0)
	    if cv2.waitKey(1) & 0xFF == ord('q'):
		break
    
cv2.destroyAllWindows()
