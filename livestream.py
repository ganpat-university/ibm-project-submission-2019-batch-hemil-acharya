
import face_recognition
import cv2
import numpy as np
from faces import known_cface_encodings, known_cface_names, known_mface_encodings, known_mface_names

camera = cv2.VideoCapture(0)

#Initializing variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def gen_frames():  
    while True:
#Reading the camera frame
        success, frame = camera.read()  
        if not success:
            break
        else:
#Resizing frame of video to 1/2 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
#Converting the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_s_frame = small_frame[:, :, ::-1]

#Only process every other frame of video to save time
                        
#Finding all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_s_frame)
            face_encodings = face_recognition.face_encodings(rgb_s_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
#Checking if the face is a match for the known face(s)
                crime_matches = face_recognition.compare_faces(known_cface_encodings, face_encoding)
                miss_matches = face_recognition.compare_faces(known_mface_encodings, face_encoding)
                name = "Unknown"
#Or using the known face with the smallest distance to the new face
                face_distance = face_recognition.face_distance(known_cface_encodings, face_encoding)
                cbest_match_index = np.argmin(face_distance)
                mface_distances = face_recognition.face_distance(known_mface_encodings, face_encoding)
                mbest_match_index = np.argmin(mface_distances)
                if crime_matches[cbest_match_index] and not miss_matches[mbest_match_index]:
                    name = known_cface_names[cbest_match_index] +" - has Criminal Record!!"
                if miss_matches[mbest_match_index] and not crime_matches[cbest_match_index]:
                    name = known_mface_names[mbest_match_index] + "- Missing Person!"
                if miss_matches[mbest_match_index] and crime_matches[cbest_match_index]:
                    name = known_mface_names[mbest_match_index] + "- Missing and has Criminal Record too!"
                face_names.append(name)
                                       
#Displaying the results
                for (top, right, bottom, left), name in zip(face_locations, face_names):
#Scaling back up face locations since the frame we detected in was scaled to 1/4 size
                    top *= 2
                    right *= 2
                    bottom *= 2
                    left *= 2                             
                               
#Drawing a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 245, 0), 2)

#Labeling with a name below the face
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 245, 0), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_TRIPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
                          
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                            
                yield (b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
cv2.waitKey(0)
cv2.destroyAllWindows()
