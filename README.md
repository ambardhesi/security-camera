security-camera

A security camera(face recognizer) for the Raspberry Pi.

To use, first ensure OpenCV 3.0.0 OpenCV_contrib modules are installed. Then, data in the form of jpeg images must be provided to train the camera. Data must be stored in format "/SecurityCamera/s1/1.jpg" where s1, s2, s3...are the different recognized people, and 1.jpg, 2.jpg.. are the different images for the people in the database.

Run face_detector.py.

The program then detects if the person in front of the camera is a known person or not.

The confidence level requires some experimenting to get right. 50 seems to work for me. 
