import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader


ROOT_FOLDER_OG = r"C:\Users\batuh\Desktop\4th_year\Thesis\data\imdb\imdb_crop\\" # Set this to wherever your imdb_crop is stored
ROOT_FOLDER_112 = r"" # can be left empty

class AgeDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.copy()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        im_path = self.root_dir + "/" + row['full_path']
        im = Image.open(im_path)
        if im.mode == 'L':
            im = im.convert('RGB')
        age = row['age']
        if self.transform:
            im = self.transform(im)
        sample = {'image':im, 'label':age}
        return sample


def get_dl(train_csv, val_csv, test_csv, test_uniform_csv, img_size, batch_size = 50, drop_last = True, num_workers=0):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    val_df = pd.read_csv(val_csv)
    test_uniform_df = pd.read_csv(test_uniform_csv)

    if(img_size == 'og'):
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        root_img_folder = ROOT_FOLDER_OG
    elif(img_size == '112'):
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Resize((112,112)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        root_img_folder = ROOT_FOLDER_112

    train_ds = AgeDataset(train_df, root_img_folder, transform)
    test_ds = AgeDataset(test_df, root_img_folder, transform)
    val_ds = AgeDataset(val_df, root_img_folder, transform)
    test_uniform_ds = AgeDataset(test_uniform_df, root_img_folder, transform)

    if(num_workers > 0):
        print(f"num_workers > 0: {num_workers}", flush=True)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers, pin_memory=True)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers, pin_memory=True)
        test_uniform_dl = DataLoader(test_uniform_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers, pin_memory=True)
    else:
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)
        test_uniform_dl = DataLoader(test_uniform_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)

    return train_dl, test_dl, val_dl, test_uniform_dl

        

