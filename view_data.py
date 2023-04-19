import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    l1_df = pd.read_csv(r"") # Insert path to test_preds.csv of a model trained with l1
    supcr_df = pd.read_csv(r"") # Insert path to test_preds.csv of a model trained with SupCR

    label_rec = []
    mae_gains = []

    for label in l1_df['acc'].unique():
        l1_sub = l1_df[l1_df['acc'] == label]
        l1_mae = abs(l1_sub['pred'] - l1_sub['acc']).mean()
        
        supcr_sub = supcr_df[supcr_df['acc'] == label]
        supcr_mae = abs(supcr_df['pred'] - supcr_df['acc']).mean()

        mae_gains.append(l1_mae - supcr_mae)
        label_rec.append(label)
    colours = ['r' if y<0 else 'g' for y in mae_gains]

    plt.subplots(figsize=(15,5))
    plt.bar(label_rec,mae_gains,color=colours,width=1)
    plt.title("MAE Gains")
    plt.xlabel("Age")
    plt.ylabel("MAE Gain")
    plt.xlim(-5,85)
    plt.show()
    
    train_csv = pd.read_csv("imdb_train.csv")
    # print(train_csv['age'].max())
    # print(train_csv['age'].min())
    plt.subplots(figsize=(15,5))
    plt.hist(train_csv["age"],76)
    plt.title("Training Distribution")
    plt.xlabel("Age")
    plt.ylabel("Num Samples")
    plt.xlim(-5,85)
    plt.show()

    train_csv = pd.read_csv("imdb_train.csv")
    # print(train_csv['age'].max())
    # print(train_csv['age'].min())
    plt.subplots(figsize=(6,4))
    plt.hist(train_csv["age"],76)
    plt.title("Training Distribution")
    plt.xlabel("Age")
    plt.ylabel("Num Samples")
    plt.xlim(-5,85)
    plt.show()


    test_csv = pd.read_csv("imdb_test.csv")
    print(test_csv['age'].max())
    print(test_csv['age'].min())
    plt.subplots(figsize=(6,4))
    plt.hist(test_csv["age"],80)
    plt.title("Test Distribution")
    plt.xlabel("Age")
    plt.ylabel("Num Samples")
    plt.xlim(-5,85)
    plt.show()