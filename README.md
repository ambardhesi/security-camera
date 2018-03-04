#security-camera

A security camera(face recognizer and motion detector) for the Raspberry Pi.

OpenCV 3.0.0 OpenCV_contrib modules must be installed to use this application. Provide jpeg images as data to train the camera and store it in format "/SecurityCamera/s1/1.jpg" where s1, s2, s3...are the different recognized people, and 1.jpg, 2.jpg.. are the different images for the people in the database.

Make the required changes in the file conf2.json, by editing the 3 Dropbox related fields. Rename the file to conf.json, and run security_camera.py with --conf "path" where path is the path to the json file.

The program detects if the person in front of the camera is a known person and if there is any motion in front of the camera.

The program will uploaded images to Dropbox if motion is detected at regular intervals.

The confidence level requires some experimenting to be accurate 
