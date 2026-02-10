# import os
# import shutil
# import pickle
# src_dirs = [
#     '/workspaces/Bollywood-Celebrity-Resemblence/data/Bollywood_celeb_face_localized/bollywood_celeb_faces_0',
#     '/workspaces/Bollywood-Celebrity-Resemblence/data/Bollywood_celeb_face_localized/bollywood_celeb_faces_1',
#     '/workspaces/Bollywood-Celebrity-Resemblence/data/Bollywood_celeb_face_localized/bollywood_celeb_faces2'
# ]
# dest_dir = '/workspaces/Bollywood-Celebrity-Resemblence/data/all_actors'
# os.makedirs(dest_dir, exist_ok=True)
# for src in src_dirs:
#     for actor in os.listdir(src):
#         src_path = os.path.join(src, actor)
#         dest_path = os.path.join(dest_dir, actor)
#         if os.path.isdir(src_path):
#             shutil.move(src_path, dest_path)

# actors = os.listdir('/workspaces/Bollywood-Celebrity-Resemblence/data/all_actors')

# print(actors)

# filenames = []

# for actor in actors:
#     for file in os.listdir(os.path.join('/workspaces/Bollywood-Celebrity-Resemblence/data/all_actors', actor)):
#         filenames.append(os.path.join('/workspaces/Bollywood-Celebrity-Resemblence/data/all_actors', actor, file))

# print(filenames)
# print(len(filenames))

# pickle.dump(filenames, open('filenames.pkl', 'wb'))

from tensorflow.keras.utils import layer_utils
from keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle

filenames = pickle.load(open('filenames.pkl', 'rb'))

model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

print(model.summary())
