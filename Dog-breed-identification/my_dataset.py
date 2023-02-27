from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import json


class My_Dataset(Dataset):
    def __init__(self, mode, transform):
        self.mode = mode
        if mode == 'test':
            self.root = './test'
        else:
            self.root = './train'
        self.df = pd.read_csv(mode + '.csv')
        self.transform = transform
        with open('./class_indies.json') as f:
            self.class_dict = json.load(f)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root,str(self.df['id'][idx]) + '.jpg')
        id = self.df['id'][idx]

        assert os.path.exists(image_path)
        img = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.mode in ['train', 'val']:
            label = self.class_dict[self.df['breed'][idx]]
            return img, label
        else:
            return img,id

    def __len__(self):
        return len(self.df)


# transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
# dataset = My_Dataset('val', transform)
# loader = DataLoader(dataset, batch_size=4)
# with open('./class_indies.json') as f:
#     class_dict = json.load(f)
# name_dict = dict((v,k) for k,v in class_dict.items())
# for data in loader:
#     img, label = data
#     for cls in label:
#         print(name_dict[cls.item()])
#     break
