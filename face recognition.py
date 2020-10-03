#16JULY 2020
#BY PRADEEP RAJ (M.P.R)
#PYTHON FACE RECOGNITION
#This Program is learned fron python face-recognition example library
#This is uploaded for THE FAUCETERS Subscribers



#import libraries
import face_recognition
import cv2
import numpy as np

#for getting a reference from webcam
video_capture = cv2.VideoCapture(0)


#load some images of you want to recognise

#you can add upto several images for loading , Just load that images in that given form 

person_1_image = face_recognition.load_image_file("\\path of person_1 image file")
person_1_face_encoding = face_recognition.face_encodings(person_1_image)[0]

person_2_image = face_recognition.load_image_file("\\path of person_2 image file")
person_2_face_encoding = face_recognition.face_encodings(person_2_image)[0]

#create arrays of known_face_encodings & names
known_face_encodings = [
    person_1_face_encoding
    person_2_face_encoding
]

known_face_names = [
    "Person 1 Name"
    "Person 2 Name"
    #Just gave a names for that loaded images  
]

face_locations = []
face_encodings = []
face_names = []


process_this_frame = True
#processing the i\p frame 
while True :
    #for grab a frame
    ret, frame = video_capture.read() 
    #resize the frame for faster recognition
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    #convert bgr colour frame into rgb colour frame for face recognition
    rgb_small_frame = small_frame[:,:,::-1]

    if process_this_frame :
        #for find the all faces and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)

        face_names = []

        for face_encoding in face_encodings :
            #match the displayed faces in the frame with known faces 
            matches = face_recognition.compare_faces(known_face_encodings,face_encoding)

            name = 'unknown'

            face_distances = face_recognition.face_distance(known_face_encodings,face_encoding)
            best_match_index = np.argmin(face_distances)

            #if the face is matched with known face append the name of the known face
            if matches[best_match_index] :
                name = known_face_names[best_match_index]
            face_names.append(name)
    
    process_this_frame = not process_this_frame

    #display the results
    for (top,right,bottom,left), name in zip(face_locations,face_names) :
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        #draw a box around the face displayed in the frame
        cv2.rectangle(frame,(left,top),(right,bottom),(255,0,0),2)
        #draw a label with a name below the face
        cv2.rectangle(frame,(left,bottom-35),(right,bottom),(255,0,0),cv2.FILLED)
        #font for the text
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame,name,(left+6,bottom-6),font,1.0,(255,255,255),1)

    #Display resulting frame 
    cv2.imshow('FAUCETER Face_Recognition',frame)

    #exit key for quit a program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()


