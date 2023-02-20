import torch
import model
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from my_dataset import My_Dataset
import json
import pandas as pd
import datetime


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Inferencing with {device}!')

    transform = transforms.Compose([
        transforms.Resize((224, 224,)),
        transforms.ToTensor(),
        transforms.Normalize((0.7589, 0.7788, 0.7599), (0.1574, 0.1497, 0.1826))
    ])

    test_dataset = My_Dataset('test', transform)
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=48)

    with open('./class_indies.json', 'r') as f:
        class_dict = json.load(f)
    name_dict = dict((v, k) for k, v in class_dict.items())
    image = []
    label = []

    weights_dict = torch.load('./resnet.pth', map_location='cpu')
    net = model.resnet50(num_class=176)
    net.load_state_dict(weights_dict)
    net.to(device)

    net.eval()
    with torch.no_grad():
        for img, path in test_loader:
            img = img.to(device)
            output = net(img)
            predict = torch.max(output, dim=1)[1]
            predict = [name_dict[x.item()] for x in predict]
            image.extend(path)
            label.extend(predict)

    df = pd.DataFrame({'image': image,
                       'label': label})
    df.to_csv(f'./submission-{datetime.date.today()}.csv', index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')

    args = parser.parse_args()
    main(args)
