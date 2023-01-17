import numpy as np

import torch
from torch.utils.data import DataLoader,SubsetRandomSampler
from model import mymodel
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

import torchvision.transforms as transforms


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tb_writer = SummaryWriter(log_dir='./logs')

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    train_data = datasets.MNIST(root='./data',train=True,transform=transform,download=True)

    # 给出一个长度为测试集的列表
    index = list(range(len(train_data)))

    # 将列表打乱
    np.random.seed(0)
    np.random.shuffle(index)

    # 给出一个划开点
    split = int(np.floor(0.8 * len(train_data)))
    train_idx,valid_idx = index[:split],index[split:]

    # 定义抽样器
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_data,batch_size=20,sampler=train_sampler)
    valid_loader = DataLoader(train_data,batch_size=20,sampler=valid_sampler)

    model = mymodel().to(device)

    # 将模型写入tensorboard
    # 先给定一个输入
    init_img = torch.zeros((20,1,28,28),device=device)
    tb_writer.add_graph(model,input_to_model=init_img)

    criterion = nn.CrossEntropyLoss()
    optimizor = torch.optim.SGD(params=model.parameters(),lr=1e-3,momentum=0.9)

    best_acc = 0.0
    for epoch in range(50):

        model.train()
        running_loss = 0.0
        for step, data in enumerate(train_loader, 0):
            images, labels = data

            # 梯度清零
            optimizor.zero_grad()
            # 前向传播，计算损失
            outputs = model(images.to(device))
            loss = criterion(outputs,labels)

            # 向后传播,更新参数
            loss.backward()
            optimizor.step()

            running_loss += loss.item()*images.shape[0]

        tb_writer.add_scalar('loss',running_loss/500,epoch)

        model.eval()
        acc = 0.0
        with torch.no_grad():
            for val_data in valid_loader:
                images,labels = val_data
                outputs = model(images.to(device))
                pred_y = torch.max(outputs,dim=1)[1]
                acc += torch.eq(pred_y,labels.to(device)).sum().item()
        acc = acc/(len(valid_loader)*20)
        tb_writer.add_scalar('acc',acc,epoch)

        print(f'loss: {running_loss/500} , acc : {acc}')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(),'./model.pth')
    tb_writer.close()
    print("Training Finished!")


if __name__ == '__main__':
    main()









