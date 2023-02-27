import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import model
from my_dataset import My_Dataset
from torch.utils.tensorboard import SummaryWriter
import time
from scheduler_warmup import CosineAnnealingLR_Warmup


def id2smooth(idx: torch.Tensor, num_classes, eps=0.1):
    """

    :param num_classes:
    :param eps:
    :param idx: tensor -> size([C])
    :return:
    """

    index = idx.unsqueeze(dim=1)
    one_hot = torch.zeros(idx.shape[0], num_classes, device=idx.device).scatter_(1, index, 1)
    smoothed = one_hot * (1 - eps) + (1 - one_hot) * eps / (num_classes - 1)
    return smoothed


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device : {device}')

    transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop((224, 224), scale=(0.5, 1), ratio=(3.0 / 4.0, 4.0 / 3.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    train_dataset = My_Dataset('train', transform['train'])
    val_dataset = My_Dataset('val', transform['val'])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=16)

    weight_path = './resnet101-pretrained.pth'
    weight_dict = torch.load(weight_path, map_location='cpu')
    net = model.resnet101(num_class=120)
    model_dict = net.state_dict()
    model_dict.update(weight_dict)
    net.load_state_dict(model_dict)
    net.to(device)

    # 是否冻结训练
    if args.freeze:
        for k, v in net.named_parameters():
            if 'fc' not in k:
                v.requires_grad = False

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR_Warmup(optimizer, total_epoch=args.epochs, eta_min=0.1 * args.lr,
                                         warmup_start_lr=0.1 * args.lr)
    loss_funcion = nn.CrossEntropyLoss()

    tb = SummaryWriter(args.logs)

    epochs = args.epochs
    best_acc = 0.0
    best_loss = torch.inf
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
            if args.label_smooth:
                label_smoothed = id2smooth(label, num_classes=120)
                loss = loss_funcion(output, label_smoothed)
            else:
                loss = loss_funcion(output, label)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            train_predict = torch.max(output, dim=1)[1]
            train_acc += torch.eq(train_predict, label).sum().item()

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        scheduler.step()
        train_acc /= len(train_dataset)

        # val
        val_acc = 0.0
        val_loss = 0.0
        net.eval()
        with torch.no_grad():
            for data in val_loader:
                img, label = data
                img, label = img.to(device), label.to(device)

                output = net(img)
                val_predict = torch.max(output, dim=1)[1]
                val_acc += torch.eq(label, val_predict).sum().item()

                val_loss += loss_funcion(output, label).item()

        val_acc /= len(val_dataset)
        val_loss /= len(val_loader)
        end_time = time.time()
        running_time = end_time - start_time

        tb.add_scalars('', {'training_loss': training_loss / steps,
                            'train_acc': train_acc,
                            'val_acc': val_acc,
                            'val_loss': val_loss,
                            'lr': lr}, epoch + 1)

        print(f'epoch : {epoch + 1}\n'
              f'    loss : {training_loss / steps}\n'
              f'    train_acc : {train_acc}\n'
              f'    val_acc : {val_acc}\n'
              f'    val_loss : {val_loss}\n'
              f'    time : {int(running_time // 60)}m {int(running_time % 60)}s')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(net.state_dict(), args.save_path)

    tb.close()
    print('Training Finished!')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    # 训练设备
    parser.add_argument('--device', default='cuda:1')

    # learning rate
    parser.add_argument('--lr', default=1e-3, type=float)

    # epochs
    parser.add_argument('--epochs', type=int, default=100)

    # weigth path to save
    parser.add_argument('--save-path', default='./resnet.pth')

    # logs
    parser.add_argument('--logs', default='./logs')

    # freeze
    parser.add_argument('--freeze', type=bool, default=False)

    # label smooth
    parser.add_argument('--label-smooth', type=bool, default=False)

    args = parser.parse_args()
    main(args)
