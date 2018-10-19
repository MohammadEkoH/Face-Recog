# train.py
import cv2, sys, numpy, os
size = 4
faces = cv2.CascadeClassifier('lib/haarcascade_frontalface_default.xml')
eyes = cv2.CascadeClassifier('lib/haarcascade_eye.xml')
noses = cv2.CascadeClassifier('lib/haarcascade_nosez.xml');
mouths = cv2.CascadeClassifier('lib/haarcascade_mouth.xml');

fn_dir = 'att_faces'
try:
    fn_name = sys.argv[1]
except:
    print("You must provide a name")
    sys.exit(0)
path = os.path.join(fn_dir, fn_name)
if not os.path.isdir(path):
    os.mkdir(path)
(im_width, im_height) = (320, 240) 
webcam = cv2.VideoCapture(0)

# Generate name for image file
pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path)
     if n[0]!='.' ]+[0])[-1] + 1

# Beginning message
print("\n\033[94mThe program will save 20 samples. \
Move your head around to increase while it runs.\033[0m\n")

# The program loops until it has 20 images of the face.
count = 0
pause = 0
count_max = 40
while count < count_max:

    # Loop until the camera is working
    rval = False
    while(not rval):
        # Put the image from the webcam into 'frame'
        (rval, frame) = webcam.read()
        if(not rval):
            print("Failed to open webcam. Trying again...")

    # Get image size
    height, width, channels = frame.shape

    # Flip frame
    frame = cv2.flip(frame, 1, 0)

    # add brightness
    alpha = 2
    beta = 50
    frame = cv2.addWeighted(frame, alpha, numpy.zeros(frame.shape, frame.dtype), 0, beta)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
    # Scale down for speed
    mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

    # Detect faces 

    # We only consider largest face 
    muka = faces.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in muka:
        face = frame[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (im_width, im_height))
        # cv2.imshow('Muka', face);
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)   
        cv2.imwrite('%s/%s.png' % (path, pin), face_resize)


        roi_gray_eye = gray[y:y+100, x:x+w] 
        mata = eyes.detectMultiScale(roi_gray_eye, 1.15, 2)
        for (mx,my,mw,mh) in mata:
            eye = frame[y:y+100, x:x+w]
            # cv2.imshow('eye', roi_gray_eye) 
            cv2.rectangle(eye, (mx,my), (mx+mw, my+mh), (255, 255, 0), 2)  
            cv2.imwrite('%s/%s.png' % (path, pin), eye)  

        roi_gray_nose = gray[y:300+300, x:x+w] 
        hidung = noses.detectMultiScale(roi_gray_nose, 1.15, 2)
        for (mx,my,mw,mh) in hidung:
            nose = frame[y:300+300, x:x+w]
            # cv2.imshow('nose', roi_gray_nose) 
            cv2.rectangle(nose, (mx,my), (mx+mw, my+mh), (255, 255, 0), 2) 
            cv2.imwrite('%s/%s.png' % (path, pin), nose)

        roi_gray_mouth = gray[y:y+h, x:x+w] 
        mulut = mouths.detectMultiScale(roi_gray_mouth, 1.15, 2)
        for (mx,my,mw,mh) in mulut:
            mouth = frame[y:y+h, x:x+w]
            # cv2.imshow('mouth', roi_gray_mouth)  
            cv2.rectangle(mouth, (mx,my), (mx+mw, my+mh), (255, 255, 0), 2) 
            cv2.imwrite('%s/%s.png' % (path, pin), mouth)

        # Remove false positives
        if(w * 6 < width or h * 6 < height):
            print("Face too small")
        else:

            # To create diversity, only save every fith detected image
            if(pause == 0):

                print("Saving training sample "+str(count+1)+"/"+str(count_max))

                # Save image file
                # cv2.imwrite('%s/%s.png' % (path, pin), face_resize)

                pin += 1
                count += 1

                pause = 1

    if(pause > 0):
        pause = (pause + 1) % 5
    cv2.imshow('OpenCV', frame)
    key = cv2.waitKey(10)
    if key == 27:
        break
