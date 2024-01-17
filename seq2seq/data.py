from torch.utils.data import Dataset, random_split
import json

max_dataset_size = 220000
train_dataset_size = 200000
valid_dataset_size = 20001

class TRANS(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        # 以只读模式打开文件
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx > max_dataset_size:
                    break
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

if __name__=='__main__':
    data = TRANS('../data/translation2019zh/translation2019zh_train.json')
    train_data, valid_data = random_split(data, [train_dataset_size, valid_dataset_size])
    test_data = TRANS('../data/translation2019zh/translation2019zh_valid.json')    
    print(f'train set size: {len(train_data)}')
    print(f'valid set size: {len(valid_data)}')
    print(f'test set size: {len(test_data)}')
    print(next(iter(train_data)))