import face_recognition as fr
import os

#Training criminal faces
cpath = "./static/Criminals/"

known_cface_names = []
known_cface_encodings = []

images = os.listdir(cpath)

for _ in images:
    image = fr.load_image_file(cpath + _)
    image_path = cpath + _
    encoding = fr.face_encodings(image)[0]

    known_cface_encodings.append(encoding)
    known_cface_names.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())

#Training missing faces
mpath = "./static/Missing/"

known_mface_names = []
known_mface_encodings = []

images = os.listdir(mpath)

for _ in images:
    image = fr.load_image_file(mpath + _)
    image_path = mpath + _
    encoding = fr.face_encodings(image)[0]

    known_mface_encodings.append(encoding)
    known_mface_names.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())