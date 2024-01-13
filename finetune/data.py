# 通过这个项目了解pytorch构建数据集、模型的一系列流程(甚至必须要熟练掌握)
from typing import Iterator
from torch.utils.data import Dataset, IterableDataset, DataLoader
from transformers import AutoTokenizer
import torch
import json

# 其实构建数据集并不难，本质上还是文件的读写操作
class AFQMC(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt') as f:
            for idx, line in enumerate(f):
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# 如果数据集非常巨大，难以一次性加载到内存中，我们也可以继承 IterableDataset 类构建迭代型数据集
class IterableAFQMC(IterableDataset):
    def __init__(self, data_file):
        self.data_file = data_file

    def __iter__(self) -> Iterator:
        with open(self.data_file, 'rt') as f:
            for line in f:
                yield json.loads(line.strip())

def collote_fn(batch_samples): # 这里应该有且只有一个参数
    batch_sentence_1, batch_sentence_2 = [], []
    batch_label = []
    for sample in batch_samples:
        batch_sentence_1.append(sample['sentence1'])
        batch_sentence_2.append(sample['sentence2'])
        batch_label.append(int(sample['label']))
    X = tokenizer( # 这里tokenizer在main函数里面指定就好，感觉非常神奇
        batch_sentence_1,
        batch_sentence_2,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ) # 返回的编码结果X和标签y被封装成一个元组返回
    y = torch.tensor(batch_label)
    return X, y


if __name__=='__main__': 
    train_data = AFQMC('../data/afqmc_public/train.json')
    valid_data = AFQMC('../data/afqmc_public/dev.json')
    print(train_data[0])
    print(train_data[1])
    print(train_data[2])
    print(train_data[3])
    # train_data_iter = IterableAFQMC('../data/afqmc_public/train.json')
    # iter_train_data = iter(train_data_iter)
    # print(next(iter_train_data))
    # print(next(iter_train_data))
    
    checkpoint = "bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collote_fn)
    batch_X, batch_y = next(iter(train_dataloader))
    print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
    print('batch_y shape:', batch_y.shape)
    print(batch_X)
    print(batch_y)
    # 这里每个样本都被处理成"[CLS] sentence1 [SEP] sentence2 [SEP]"的格式
    # 注意101和102分别是[CLS]和[SEP]对应的token id
