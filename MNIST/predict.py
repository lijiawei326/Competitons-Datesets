import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model import mymodel


def main():
    devcie = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.1307, ], [0.3081, ])])

    test_data = datasets.MNIST(root='./data',train=False,transform=transform,download=True)

    test_loader = DataLoader(test_data,batch_size=20,shuffle=True)

    model = mymodel().to(device=devcie)
    model.load_state_dict(torch.load('./model.pth',map_location=devcie))
    model.eval()
    acc = 0.0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data

            output = model(images.to(devcie))
            pred_y = torch.max(output,dim=1)[1]
            acc += torch.eq(pred_y,labels).sum().item()

    acc = acc / len(test_data)
    print(acc)


if __name__ == '__main__':
    main()