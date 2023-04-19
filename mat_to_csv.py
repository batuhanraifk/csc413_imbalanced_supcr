from scipy.io import loadmat
from datetime import datetime
import pandas as pd


MAT_PATH = "wiki.mat"

mat = loadmat(MAT_PATH)


dob_rec = []
photo_taken_rec = []
full_path_rec = []
gender_rec = []
name_rec = []
face_score_rec = []
second_face_score_rec = []


# --- DOB ---
for dob in mat['wiki'][0][0][0][0]:
    dob_rec.append(datetime.fromordinal(dob))
# ---

print('done dob')

# --- photo taken ---
photo_taken_rec = mat['wiki'][0][0][1][0]
# ---

# --- full_path ---
for fp in mat['wiki'][0][0][2][0]:
    full_path_rec.append(fp[0])
# ---

print('done fp')

# --- gender ---
gender_rec = mat['wiki'][0][0][3][0]
# ---

# --- name ---
for name in mat['wiki'][0][0][4][0]:
    if name:
        name_rec.append(name[0])
    else:
        name_rec.append('')
# ---

print('done names')

# --- face_score ---
face_score_rec = mat['wiki'][0][0][6][0]
# ---

# --- second_face_score ---
second_face_score_rec = mat['wiki'][0][0][7][0]
# ---

df = pd.DataFrame({
    'dob' : dob_rec,
    'photo_taken' : photo_taken_rec,
    'full_path' : full_path_rec,
    'gender' : gender_rec,
    'name' : name_rec,
    'face_score' : face_score_rec,
    'second_face_score' : second_face_score_rec
})

df.to_csv("wiki.csv")