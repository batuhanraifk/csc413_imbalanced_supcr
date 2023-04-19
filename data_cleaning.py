import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tqdm


# Set this to wherever your imdb_crop folder is 
ROOT_FOLDER = r""


def add_size(df):
    size_rec = []
    for i,row in df.iterrows():
        im = Image.open(ROOT_FOLDER + "\\" + row['full_path'])
        size_rec.append(im.size)
        if i%1000 == 0:
            print(i)
    df['size'] = size_rec
    return df

def add_age(df):
    df['age'] = df['photo_taken'] - df['dob'].str[:4].astype(int)
    return df

if __name__ == '__main__':
    df = pd.read_csv("imdb.csv")
    print(len(df))
    df = df[df['face_score'] > 2]
    print(len(df))
    df = df[df['second_face_score'].isna()]
    print(len(df))
    df = add_size(df)
    print(len(df))
    df = add_age(df)
    df = df[df['age'] > 0]
    df = df[df['age'] < 101]
    df.reset_index(inplace=True)
    df.to_csv("cleaned_imdb.csv")
        