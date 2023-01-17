from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd


class MyDataset(Dataset):
    def __init__(self, df,train:bool):
        self.data = torch.from_numpy(np.array(df)).type(torch.float32)
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.train:
            return self.data[idx][:-1],self.data[idx][-1]
        else:
            return self.data[idx]


def class_decompose(input_):
    if not pd.isna(input_):
        output = str(input_).split(',')[0].replace(' ', '')
    else:
        return np.nan
    return output


def df_class_decompose(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(class_decompose)
    return df


def make_nan(series):
    if series.dtype == 'object':
        # 统计各value的个数
        count = pd.value_counts(series, dropna=False)
        small_class = count[count < 100].index
        # 将value个数小于某值的赋值zero
        series = series.apply(lambda x: np.nan if x in small_class else x)
    return series


def df_preprocess(df):
    df = df_class_decompose(df)
    df = df.apply(make_nan)
    df = pd.get_dummies(df, dummy_na=True)
    # 去掉全零的列
    df.drop(list(df.columns[(df == 0).all().values]), axis=1, inplace=True)
    # 计算每列的均值
    col_mean = df.mean(axis=0)
    df.fillna(col_mean, inplace=True)
    return df


def get_data(train:bool):
    date_class = ['Listed On', 'Last Sold On']
    drop_col = ['Address', 'Summary', 'Elementary School Score', 'Middle School', 'High School', 'Zip'] + date_class
    nan_list = ['Unknow']

    # 数据读取
    train_df = pd.read_csv('./train.csv', index_col='Id', na_values=nan_list)
    label = train_df['Sold Price']
    train_df = train_df.drop('Sold Price', axis=1)
    test_df = pd.read_csv('./test.csv', index_col='Id', na_values=nan_list)
    # 将label去掉拼接便于预处理
    all_df = pd.concat([train_df, test_df]).drop(drop_col, axis=1)
    train_num = train_df.shape[0]

    # 对DataFrame预处理
    all_df = df_preprocess(all_df)

    if train:
        return all_df.iloc[:train_num],label
    else:
        return all_df.iloc[train_num:]