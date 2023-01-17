import torch
import os
from model import MLP
from my_dataset import MyDataset, get_data
from torch.utils.data import DataLoader
import pandas as pd
import datetime


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    test_df = get_data(train=False)
    test_dataset = MyDataset(test_df, train=False)
    test_loader = DataLoader(test_dataset, batch_size=50000)

    weight_path = './mlp.pth'
    assert os.path.exists(weight_path)
    weights_dict = torch.load(weight_path, map_location='cpu')

    net = MLP()
    net.load_state_dict(weights_dict)
    net.to(device)
    net.eval()

    Id = test_df.index.tolist()
    price = []
    for data in test_loader:
        with torch.no_grad():
            price.extend(net(data.to(device)).squeeze().tolist())

    df = pd.DataFrame({'Id': Id,
                       'Sold Price': price}, index=Id)

    df.to_csv(f'./submission-{str(datetime.date.today())}.csv', index=False)


if __name__ == '__main__':
    main()
