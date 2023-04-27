from fileinput import filename
import os
import sys
from PIL import Image
import face_recognition
import cv2
import numpy as np
from flask import Flask, render_template, flash, request, Response, url_for
from werkzeug.utils import secure_filename
from livestream import gen_frames
from faces import known_cface_encodings, known_cface_names, known_mface_encodings, known_mface_names



app = Flask(__name__)
app.secret_key = 'abc'

VID_UPLOAD_FOLDER = 'upload_vid/'
EXTENSIONS_VID = set(['txt','mp4'])
EXTENSIONS_IMG = set(['png','jpg','jpeg'])
CRIMINAL_FOLDER = 'static/Criminals/'
MISSING_FOLDER = 'static/Missing/'
app.config['CRIMINAL_FOLDER'] = CRIMINAL_FOLDER
app.config['MISSING_FOLDER'] = MISSING_FOLDER
TEST_FOLDER = 'static/Test/'
app.config['TEST_FOLDER'] = TEST_FOLDER
app.config['VID_UPLOAD_FOLDER'] = VID_UPLOAD_FOLDER

#Chunk for the uploading part
def update_dataset(upload_folder):
    if request.method == 'POST':
        pic = request.files['pic']
        if not pic:
            flash('No pic uploaded! Upload a file and try again')
            return render_template('upload.html')
        

        def allowed_file(filename):
            return '.' in filename and filename.rsplit('.', 1)[1] in EXTENSIONS_IMG
        
        if pic and allowed_file(pic.filename):
            filename = secure_filename(pic.filename)
            pic.save(os.path.join(app.config[upload_folder], filename))
                
#if loop for an unclear image        
        if not allowed_file(pic.filename):
            flash('Bad upload!')
            return render_template('upload.html')
        

        if not multiface_check(os.path.join(app.config[upload_folder], filename)):
            flash('Image uploaded')
    return render_template('upload.html')

#To check if the image uploaded contains only one person in the frame
def multiface_check(filename):
            img = Image.open(filename).convert('RGB')
            image_arr = np.array(img)
            face_locations = face_recognition.face_locations(image_arr)
            face_encodings = face_recognition.face_encodings(image_arr, face_locations)
            if len(face_encodings) > 1:
                flash('Image has more than one faces. Bad upload! Try uploading another image file.')
                os.remove(filename)
                return True
            return False

#chunk to delete the image uploaded
def delete_file(upload_folder):
    if request.method == 'POST':
        pic = request.files['pic']
        file_path = os.path.join(upload_folder, pic.filename)
        if not pic:
            flash('No file selected!')
            return render_template('delete_files.html')
        
        elif os.path.exists(file_path):
            os.remove(file_path)
            flash('File deleted successfully')
            return render_template('delete_files.html')
        
        flash('File doesnot exist!')
    return render_template('delete_files.html')
    

#Chunk for comapring the faces and checking if they match
def compare_faces(face_encoding):
    crime_matches = face_recognition.compare_faces(known_cface_encodings, face_encoding)
    miss_matches = face_recognition.compare_faces(known_mface_encodings, face_encoding)
    name = "Unknown"
#use the known face with the smallest distance to the new face
    face_distance = face_recognition.face_distance(known_cface_encodings, face_encoding)
    cbest_match_index = np.argmin(face_distance)
    mface_distances = face_recognition.face_distance(known_mface_encodings, face_encoding)
    mbest_match_index = np.argmin(mface_distances)
#Matching the image for three categories - missing/ criminal/ both 
    if crime_matches[cbest_match_index] and not miss_matches[mbest_match_index]:
        name = known_cface_names[cbest_match_index] +" - has Criminal Record!"
    if miss_matches[mbest_match_index] and not crime_matches[cbest_match_index]:
        name = known_mface_names[mbest_match_index] + "- Missing Person!"
    if miss_matches[mbest_match_index] and crime_matches[cbest_match_index]:
        name = known_mface_names[mbest_match_index] + "- Missing and has Criminal Record too!"

    return name

#providing the direction towards other files
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/criminal_rec', methods=['GET', 'POST'])
def get_criminal():
    return update_dataset('CRIMINAL_FOLDER')
    
    
@app.route('/missing_rec', methods=['GET', 'POST'])
def get_missing():
    return update_dataset('MISSING_FOLDER')

@app.route('/delete', methods=['GET', 'POST'])
def delete():
    return render_template('delete.html')

@app.route('/deletefile_from_missing_rec', methods=['GET', 'POST'])
def delete_missing():
    return delete_file(MISSING_FOLDER)

@app.route('/deletefile_from_criminal_rec', methods=['GET', 'POST'])
def delete_criminal():
    return delete_file(CRIMINAL_FOLDER)

@app.route('/image_check_get', methods=['GET', 'POST'])
def get_test():
    file_name = 'test.jpeg'
    if request.method == 'POST':
        pic = request.files['pic']
        if not pic:
            flash('No pic uploaded! Upload a file and try again')
            return render_template('upload_img.html')
        def allowed_file(filename):
            return '.' in filename and filename.rsplit('.', 1)[1] in EXTENSIONS_IMG
        
        if pic and allowed_file(pic.filename):
            filename = secure_filename(pic.filename)
            pic.save(os.path.join(app.config['TEST_FOLDER'], file_name))       
        
        if not allowed_file(pic.filename):
            flash('Bad upload!')
            return render_template('upload_img.html')

        flash('Image uploaded')
    return render_template('upload_img.html')

@app.route('/image_check')
def img_check():
    test_image = "static/Test/test.jpeg"
    image = Image.open(test_image)
    image_arr = np.array(image)
    img_cv = cv2.imread(test_image)
    face_locations = face_recognition.face_locations(image_arr)

    face_encodings = face_recognition.face_encodings(image_arr, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = compare_faces(face_encoding)
        cv2.rectangle(img_cv, (left, top), (int(right), int(bottom)), (0, 0, 255), 2)
        cv2.rectangle(img_cv, (left, int(bottom) - 15), (int(right), int(bottom)), (0, 0, 255), cv2.FILLED)

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img_cv, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
    
    cv2.imwrite("static/Test/output.jpg", img_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return render_template('img_output.html')

@app.route('/video_feed')
def video_template():
    return render_template('videofeed_output.html')   

@app.route('/video_feed_live')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_rec', methods=['GET', 'POST'])
def video_upload():
    if request.method == 'POST':
        file = request.files['file']
        def allowed_file(filename):
            return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS_VID
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['VID_UPLOAD_FOLDER'], filename))
            NEWPATH=videotest(os.path.join(app.config['VID_UPLOAD_FOLDER'], filename))
            return render_template('video.html')

    return render_template('upload_vid.html')


#Chunk for video face detection and matching
def videotest(filename):
    video_capture = cv2.VideoCapture(filename)
    length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    
    width  = int(video_capture.get(3)) 
    height = int(video_capture.get(4))

    fourcc = cv2.VideoWriter_fourcc(*'vp80')
    PATH = 'static/demo.webm'
    out = cv2.VideoWriter(PATH,fourcc, fps, (width, height))
        
    for i in range(1,length-1):
            
        ret, frame = video_capture.read()
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            
            name = compare_faces(face_encoding)
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 10), (right, bottom + 10 ), (10, 10, 10), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 2, bottom), font, 0.4, (255, 255, 255), 1)

        print()
        sys.stdout.write(f"writing...{int((i/length)*100)+1}%")
        sys.stdout.flush()
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#When everything is done, release the capture
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()
    return PATH


if __name__ == "__main__":
    app.run(debug = True)