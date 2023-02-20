import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import model
from my_dataset import My_Dataset
from torch.utils.tensorboard import SummaryWriter
import time


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device : {device}')

    transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop((224, 224), scale=(0.8, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.7589, 0.7788, 0.7599), (0.1574, 0.1497, 0.1826))
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.7589, 0.7788, 0.7599), (0.1574, 0.1497, 0.1826))
        ])
    }
    train_dataset = My_Dataset('train', transform['train'])
    val_dataset = My_Dataset('val', transform['val'])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=48)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=48)

    weight_path = './resnet18-pretrained.pth'
    pretrained_dict = torch.load(weight_path, map_location='cpu')
    del pretrained_dict['fc.weight']
    del pretrained_dict['fc.bias']

    net = model.resnet18(num_class=176)
    model_dict = net.state_dict()
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    net.to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    loss_funcion = nn.CrossEntropyLoss()

    tb = SummaryWriter(args.logs)

    epochs = args.epochs
    best_acc = 0.0
    steps = len(train_loader)
    for epoch in range(epochs):
        start_time = time.time()
        training_loss = 0.0

        # train
        train_acc = 0.0
        net.train()
        for step, data in enumerate(train_loader):
            img, label = data
            img, label = img.to(device), label.to(device)

            # 梯度清零
            optimizer.zero_grad()

            output = net(img)
            loss = loss_funcion(output, label)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            train_predict = torch.max(output,dim=1)[1]
            train_acc += torch.eq(train_predict,label).sum().item()
        train_acc /= len(train_dataset)
        # val
        val_acc = 0.0
        net.eval()
        with torch.no_grad():
            for data in val_loader:
                img, label = data
                img, label = img.to(device), label.to(device)

                output = net(img)
                val_predict = torch.max(output, dim=1)[1]
                val_acc += torch.eq(label, val_predict).sum().item()

        val_acc /= len(val_dataset)

        end_time = time.time()
        running_time = end_time - start_time

        tb.add_scalars('', {'training_loss': training_loss / steps,
                            'train_acc': train_acc,
                            'val_acc': val_acc}, epoch + 1)

        print(f'epoch : {epoch + 1}\n'
              f'    loss : {training_loss / steps}\n'
              f'    train_acc : {train_acc}\n'
              f'    val_acc : {val_acc}\n'
              f'    time : {int(running_time // 60)}m {int(running_time % 60)}s')

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), args.save_path)

    tb.close()
    print('Training Finished!')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    # 训练设备
    parser.add_argument('--device', default='cuda:0')

    # learning rate
    parser.add_argument('--lr', default=3e-2,type=float)

    # epochs
    parser.add_argument('--epochs', default=50)

    # weigth path to save
    parser.add_argument('--save-path', default='./resnet.pth')

    # logs
    parser.add_argument('--logs', default='./logs')

    args = parser.parse_args()
    main(args)
