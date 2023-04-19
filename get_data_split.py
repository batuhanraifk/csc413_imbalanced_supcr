import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


SEED = 0

# def get_bimodal(df):
def gen_bimodal(desired_dataset_size = 20000):
    N=int(desired_dataset_size/2)
    mu, sigma = 20, 7
    mu2, sigma2 = 60, 7
    X1 = np.random.normal(mu, sigma, N)
    X2 = np.random.normal(mu2, sigma2, N)
    X = np.concatenate([X1, X2])
    X = X.astype(int)
    plt.hist(X, bins = 100)
    plt.title("Target data distribution:")
    plt.show()
    unique_ages = np.unique(X)
    age_count_dict = dict([])
    for age in unique_ages:
        age_count_dict[age] = (X == age).sum()
    return age_count_dict


def get_num_data_dict(df):
    unique_age = df['age'].unique()
    age_count_dict = dict([])
    for age in unique_age:
        age_count_dict[age] = (df['age'] == age).sum()
    return age_count_dict



def check_counts(desired_count, have_count):
    ages = desired_count.keys()
    for age in ages:
        if age <= 0:
            continue
        if have_count[age] < desired_count[age]:
            print(age, have_count[age], desired_count[age])
            return False
    return True

def gen_train_data(df, size):
    want_count = gen_bimodal(size)
    have_count = get_num_data_dict(df)
    print("Checking counts:")
    print(check_counts(want_count, have_count))

    SKIP_FRQ = 0.05
    skip_next_cnt = 0
    df_rec = []
    for age in want_count:
        skip = np.random.random()
        if (skip_next_cnt >= 6):
            skip_next_cnt = 0
        
        if (skip < SKIP_FRQ) or ((skip_next_cnt > 0) and (skip_next_cnt < 6)):
            skip_next_cnt += 1
            continue

        if (age > 35) and (age < 45):
            continue
        if (age <= 0) or (age >= 100):
            continue
    
        age_subset = df[df['age'] == age].copy()
        age_subset.reset_index(drop=True, inplace=True)
        a = shuffle(age_subset,random_state=SEED)
        df_rec.append(a.head(want_count[age]))
    train_df = pd.concat(df_rec, ignore_index=True)
    return train_df

def gen_test_data(df, train_df, size = 5000):
    used_imgs = set(train_df['full_path'])
    unused_df = df[~df['full_path'].isin(used_imgs)]

    df_rec = []
    want_count = gen_bimodal(size)
    have_count = get_num_data_dict(df)
    for age in want_count:
        if (age <= 0) or (age >= 100):
            continue
    
        age_subset = unused_df[unused_df['age'] == age].copy()
        age_subset.reset_index(drop=True, inplace=True)
        a = shuffle(age_subset,random_state=SEED)
        df_rec.append(a.head(want_count[age]))
    test_df = pd.concat(df_rec, ignore_index=True)
    return test_df

def gen_uniform_test_data(start_age, end_age, per_age_img, df, train_df, test_df):
    used_imgs = set(train_df['full_path'])
    used_imgs.update(test_df['full_path'])

    unused_df = df[~df['full_path'].isin(used_imgs)]

    df_rec = []
    for age in range(100):
        if ((age < start_age) or (age > end_age)):
            age_subset = unused_df[unused_df['age'] == age].copy()
            age_subset.reset_index(drop=True, inplace=True)
            df_rec.append(age_subset)
            continue

        age_subset = unused_df[unused_df['age'] == age].copy()
        age_subset.reset_index(drop=True, inplace=True)
        a = shuffle(age_subset,random_state=SEED)
        df_rec.append(a.head(per_age_img))
    uniform_test_df = pd.concat(df_rec, ignore_index=True)
    return uniform_test_df


def main_imbalanced():
    TRAIN_SET_SIZE = 5000
    TEST_SET_SIZE = 1000

    UNIFORM_START_AGE = 1
    UNIFORM_END_AGE = 100
    PER_AGE_IMG = 10
    # IN total, will have roughly (UNIFORM_END_AGE - UNIFORM_START_AGE)*PER_AGE_IMG # of photos

    df = pd.read_csv("cleaned_imdb.csv")
    train_df = gen_train_data(df, TRAIN_SET_SIZE)
    train_df, val_df = train_test_split(train_df, test_size=0.1)
    test_df = gen_test_data(df, train_df, TEST_SET_SIZE)
    uniform_test_df = gen_uniform_test_data(UNIFORM_START_AGE, UNIFORM_END_AGE, PER_AGE_IMG, df, train_df, test_df)

    train_imgs = set(train_df['full_path'])
    test_imgs = set(test_df['full_path'])
    uniform_test_imgs = set(uniform_test_df['full_path'])

    if (train_imgs & test_imgs):
        print("Train and test have imgs in common!")
    if (train_imgs & uniform_test_imgs):
        print("Train and uniform_test have imgs in common!")
    if (test_imgs & uniform_test_imgs):
        print("Test and uniform have imgs in common!")

    plt.hist(train_df['age'],90)
    plt.title("Train distribution")
    plt.show()
    plt.hist(val_df['age'],90)
    plt.title("Val distribution")
    plt.show()
    plt.hist(test_df['age'],90)
    plt.title("Test distribution")
    plt.show()
    plt.hist(uniform_test_df['age'],90)
    plt.title("Uniform Test distribution")
    plt.show()
    # train_df.to_csv("imdb_train.csv")
    # val_df.to_csv("imdb_val.csv")
    # test_df.to_csv("imdb_test.csv")
    # uniform_test_df.to_csv("imdb_test_uniform.csv")


def main_original_distribution(subsample=False, seed=0):
    # Does not produce uniform test

    df = pd.read_csv("cleaned_imdb.csv")
    if subsample:
        df,_ = train_test_split(df, train_size=subsample,random_state=seed)
    train_val_df, test_df = train_test_split(df, train_size=0.8,random_state=seed)
    train_df, val_df = train_test_split(train_val_df, test_size=0.125,random_state=seed)
    print(train_df.shape)
    print(test_df.shape)
    print(val_df.shape)
    
    train_imgs = set(train_df['full_path'])
    val_imgs = set(val_df['full_path'])
    test_imgs = set(test_df['full_path'])

    if (train_imgs & test_imgs):
        print("Train and test have imgs in common!")
    if (train_imgs & val_imgs):
        print("Train and uniform_test have imgs in common!")
    if (test_imgs & val_imgs):
        print("Test and uniform have imgs in common!")

    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True,inplace=True)
    test_df.reset_index(drop=True,inplace=True)

    plt.hist(train_df['age'],90)
    plt.title("Train distribution")
    plt.show()
    plt.hist(val_df['age'],90)
    plt.title("Val distribution")
    plt.show()
    plt.hist(test_df['age'],90)
    plt.title("Test distribution")
    plt.show()
    train_df.to_csv("imdb_train.csv")
    val_df.to_csv("imdb_val.csv")
    test_df.to_csv("imdb_test.csv")
    

if __name__ == '__main__':
    df = pd.read_csv("cleaned_imdb.csv")
    plt.hist(df['age'],100)
    plt.title("Original distribution")
    plt.show()
    np.random.seed(SEED)
    main_imbalanced()