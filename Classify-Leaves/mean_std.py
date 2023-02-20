import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
from PIL import Image


class mydataset(Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.root = root
        self.file_list = os.listdir(root)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.file_list[idx])

        assert os.path.exists(img_path)
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.file_list)


transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor()])
dataset = mydataset('./images',transform)
loader = DataLoader(dataset,batch_size=1)
mean = torch.zeros(3)
std = torch.zeros(3)
for data in loader:
    for d in range(3):
        mean[d] += data[:,d,:,:].mean()
        std[d] += data[:,d,:,:].std()

mean.div_(len(dataset))
std.div_(len(dataset))
print(mean,std)
