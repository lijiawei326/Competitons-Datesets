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
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = My_Dataset('test', transform)
    test_loader = DataLoader(test_dataset, batch_size=128, num_workers=16)

    with open('./class_indies.json', 'r') as f:
        class_dict = json.load(f)

    weights_dict = torch.load('./resnet-pre3.pth', map_location='cpu')
    net = model.resnet50(num_class=120)
    net.load_state_dict(weights_dict)
    net.to(device)

    ids = []
    samples = []
    net.eval()
    with torch.no_grad():
        for img, id in test_loader:
            img = img.to(device)
            output = net(img)
            predict = torch.softmax(output, dim=1)
            samples.extend(predict.tolist())
            ids.extend(id)

    df = pd.DataFrame(samples, columns=class_dict.keys(), index=ids)
    df_sample_submission = pd.read_csv('test.csv', index_col=0)
    df = df[df_sample_submission.columns.tolist()]
    df.to_csv(f'./submission-{datetime.date.today()}.csv',index_label='id')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')

    args = parser.parse_args()
    main(args)
