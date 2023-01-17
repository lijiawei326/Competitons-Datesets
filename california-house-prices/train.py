import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from my_dataset import MyDataset
import my_dataset
from torch.utils.data import DataLoader
import random
from model import MLP
from torch.utils.tensorboard import SummaryWriter





def split_set(_train_df, validate_rate):
    val_index = random.sample(range(len(_train_df)), k=int(len(_train_df) * validate_rate))
    val_df = _train_df.iloc[val_index]
    _train_df = _train_df.drop(val_index)
    return _train_df, val_df


def log_rmse(output, label):
    output = torch.clamp(output,1,float('inf'))
    rmse = torch.sqrt(loss_function(torch.log(output), torch.log(label)))
    return rmse


if __name__ == '__main__':
    devcie = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Start training with {devcie}')

    # 获取训练数据
    train_df,label = my_dataset.get_data(train=True)

    # 将预处理好的train_df加上标签
    train_df = pd.concat((train_df, label), axis=1)

    # 将DataFrame划分为训练集与验证集
    val_rate = 0.2
    train_df, val_df = split_set(train_df, val_rate)

    trian_dataset = MyDataset(train_df,train=True)
    val_dataset = MyDataset(val_df,train=True)

    num_workers = 48
    batch_size = 50000
    train_loader = DataLoader(trian_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)

    net = MLP()
    net.to(devcie)
    optimizer = torch.optim.SGD(params=net.parameters(), lr=1e-2, weight_decay=0.1)
    loss_function = nn.MSELoss()

    tb_writer = SummaryWriter(log_dir='./logs')
    epochs = 300
    steps = len(train_loader)
    save_path = './mlp.pth'
    best_val_loss = np.inf
    for epoch in range(epochs):
        running_loss = 0.0

        net.train()
        for step, data in enumerate(train_loader):
            input, label = data
            input,label = input.to(devcie),label.to(devcie)

            # 梯度清零，前向传播
            optimizer.zero_grad()
            output = net(input).squeeze()

            # 计算损失，后向传播
            loss = log_rmse(output, label)
            loss.backward()

            # 更新参数
            optimizer.step()

            running_loss += loss.item()

        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                input, label = data
                input, label = input.to(devcie), label.to(devcie)
                output = net(input).squeeze()
                val_loss += log_rmse(output, label).item()

        tb_writer.add_scalars('loss', {'train': running_loss / steps,
                                       'val': val_loss / len(val_loader)}, epoch + 1)

        print(f'epoch : {epoch + 1}\n'
              f'       train_loss : {running_loss / steps}\n'
              f'       val_loss   : {val_loss / len(val_loader)}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), save_path)

    tb_writer.close()
    print('Trainging Finished!')
