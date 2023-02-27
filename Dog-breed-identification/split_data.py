import pandas as pd
import random
import json


def main():
    df = pd.read_csv('./labels.csv')
    labels = pd.unique(df['breed'])

    # 将标签名称转化为数字，并写入json文件
    class_dict = dict((k, v) for v, k in enumerate(labels))
    json_str = json.dumps(class_dict, indent=4)
    with open('class_indies.json', 'w') as j:
        j.write(json_str)

    # 随机抽取样本分为测试集与验证集
    val_rate = 0.2
    val_indices = []

    # 从每个类别中抽取20%的数据，保证数据分布的一致性
    for l in labels:
        index = df[df['breed'] == l].index.values
        val_index = random.sample(sorted(index), k=int(len(index) * val_rate))
        val_indices.extend(val_index)

    random.shuffle(val_indices)
    train_indices = df.index.values
    train_indices = [x for x in train_indices if x not in val_indices]
    random.shuffle(train_indices)

    val_df = df.iloc[val_indices]
    train_df = df.iloc[train_indices]

    val_df.to_csv('val.csv', index=False)
    train_df.to_csv('train.csv', index=False)


if __name__ == '__main__':
    main()
