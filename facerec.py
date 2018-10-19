# facerec.py
import cv2, sys, numpy, os
size = 2
haar_face = 'lib/haarcascade_frontalface_default.xml'
haar_eye = "lib/haarcascade_eye.xml"
haar_nose = "lib/haarcascade_nosez.xml"
haar_mouth = "lib/haarcascade_mouth.xml"
fn_dir = 'att_faces'
font_besar = cv2.FONT_HERSHEY_SIMPLEX
font_kecil = cv2.FONT_HERSHEY_PLAIN

# Part 1: Create fisherRecognizer
print('Training...')

# Create a list of images and a list of corresponding names
(images, lables, names, id) = ([], [], {}, 0) 

# Get the folders containing the training data
for (subdirs, dirs, files) in os.walk(fn_dir):
    
    # Loop through each folder named after the subject in the photos
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(fn_dir, subdir)
 
        # Loop through each photo in the folder
        for filename in os.listdir(subjectpath):
 
            # Skip non-image formates 
            f_name, f_extension = os.path.splitext(filename)
            if(f_extension.lower() not in
                    ['.png','.jpg','.jpeg','.gif','.pgm']): 
                print("Skipping "+filename+", wrong file type")
                continue  
            path = subjectpath + '/' + filename
            lable = id

            # Add to training data
            images.append(cv2.imread(path, 0))
            lables.append(int(lable))
        id += 1 
(im_width, im_height) = (320, 240)

# Create a Numpy array from the two lists above
(images, lables) = [numpy.array(lis) for lis in [images, lables]]


# OpenCV trains a model from the images
# NOTE FOR OpenCV2: remove '.face'
prediksi_face = cv2.face.LBPHFaceRecognizer_create() 
prediksi_face.train(images, lables)  

# eyes = cv2.eye.LBPHFaceRecognizer_create()
# eyes.train(images, lables)

# Part 2: Use fisherRecognizer on camera stream
haar_cascade_face = cv2.CascadeClassifier(haar_face)
haar_cascade_eye = cv2.CascadeClassifier(haar_eye)
haar_cascade_nose = cv2.CascadeClassifier(haar_nose)
haar_cascade_mouth = cv2.CascadeClassifier(haar_mouth)

webcam = cv2.VideoCapture(0)

while True:
 
    # Loop until the camera is working
    rval = False
    while(not rval):
        # Put the image from the webcam into 'frame'
        (rval, frame) = webcam.read()
        if(not rval):
            print("Failed to open webcam. Trying again...")
 
    # Flip the image (optional)
    frame=cv2.flip(frame,1,0) 

    alpha = 2
    beta = 60
    frame = cv2.addWeighted(frame, alpha, numpy.zeros(frame.shape, frame.dtype), 0, beta)

    # Convert to grayscalel
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize to speed up detection (optinal, change size above)
    mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size))) 

    # Detect faces and loop through each one
    faces = haar_cascade_face.detectMultiScale(mini)
    for i in range(len(faces)):
        face_i = faces[i]

        # Coordinates of face after scaling back by `size`
        (x, y, w, h) = [v * size for v in face_i]
        face = gray[y:y + h, x:x + w] 
        cv2.imshow('face', face) 
        face_resize = cv2.resize(face, (im_width, im_height))
        # Try to recognize the face
        prediction_face = prediksi_face.predict(face_resize)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3) 
        cv2.putText(frame, '%s - %.0f' % (names[prediction_face[0]],prediction_face[1]), (x-10, y-10), font_besar,1,(0, 255, 0))
 
        eyes = haar_cascade_eye.detectMultiScale(mini) 
        for j in range(len(eyes)):
            eye_j = eyes[j]

            eye = gray[y:y + h, x:x + w] 
            cv2.imshow('eye', eye) 
            (x, y, w, h) = [v * size for v in eye_j]
            eye_resize = cv2.resize(eye, (im_width, im_height)) 
            prediction = prediksi_face.predict(eye_resize) 
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3) 
            cv2.putText(frame, '%s - %.0f' % (names[prediction[0]],prediction[1]), (x-10, y-10), font_kecil ,1,(255, 255, 0))
        
        noses = haar_cascade_nose.detectMultiScale(mini) 
        for j in range(len(noses)):
            noses_j = noses[j]

            nose = gray[y:y + h, x:x + w] 
            cv2.imshow('nose', nose) 
            (x, y, w, h) = [v * size for v in noses_j] 
            nose_resize = cv2.resize(nose, (im_width, im_height))
            prediction = prediksi_face.predict(nose_resize) 
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
            cv2.putText(frame, '%s - %.0f' % (names[prediction[0]],prediction[1]), (x-10, y-10), font_kecil ,1,(255, 255, 0))
 
        mouths = haar_cascade_mouth.detectMultiScale(mini) 
        for j in range(len(mouths)):
            mouths_j = mouths[j]
            
            mouth = gray[y:y + h, x:x + w] 
            cv2.imshow('mouth', mouth) 
            (x, y, w, h) = [v * size for v in mouths_j]
            mouth_resize = cv2.resize(mouth, (im_width, im_height)) 
            prediction = prediksi_face.predict(mouth_resize) 
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
            cv2.putText(frame, '%s - %.0f' % (names[prediction[0]],prediction[1]), (x-10, y-10), font_kecil ,1,(255, 255, 0))
 
 
    # Show the image and check for ESC being pressed
    cv2.imshow('OpenCV', frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
