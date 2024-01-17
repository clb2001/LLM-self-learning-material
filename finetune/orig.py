# 通过这个项目了解pytorch构建数据集、模型的一系列流程(甚至必须要熟练掌握)
from typing import Iterator
from torch.utils.data import Dataset, IterableDataset, DataLoader
from transformers import AutoTokenizer
import torch
import json
from torch import nn
from transformers import BertPreTrainedModel, BertModel
from transformers import AutoConfig
from tqdm.auto import tqdm
from transformers import AdamW, get_scheduler
# 随机种子操作设置
import random
import os
import numpy as np

def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

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

class BertForPairwiseCLS(BertPreTrainedModel):
    def __init__(self, config): # 这里应该有且只有一个参数
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False) # 这里命名要和原来保持一致，必须是self.bert
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(768, 2)
        self.post_init()
        
    def forward(self, x):
        bert_output = self.bert(**x)
        cls_vectors = bert_output.last_hidden_state[:, 0, :]
        cls_vectors = self.dropout(cls_vectors)
        logits = self.classifier(cls_vectors)
        return logits

def train_loop(data_loader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(data_loader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_step_num = (epoch - 1) * len(data_loader)
    model.train()
    for step, (X, y) in enumerate(data_loader, start=1):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad() # 将优化器的梯度清零，以便进行下一步的反向传播。
        loss.backward()
        optimizer.step()
        lr_scheduler.step() # 更新学习率。
        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss / (finish_step_num + step):>7f}')
        progress_bar.update(1) # 更新进度条，表示完成了一步训练。
    return total_loss
        
def test_loop(dataloader, model, mode='test'):
    assert mode in ['Test', 'Valid']
    size = len(dataloader.dataset)
    correct = 0
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    correct /= size
    print(f"{mode} Accuracy: {(100 * correct):>0.1f}%\n")
    return correct  


if __name__=='__main__': 
    train_data = AFQMC('../data/afqmc_public/train.json')
    valid_data = AFQMC('../data/afqmc_public/dev.json')
    checkpoint = "bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    train_dataloader = DataLoader(train_data, batch_size=6, shuffle=True, collate_fn=collote_fn)
    valid_dataloader = DataLoader(valid_data, batch_size=6, shuffle=True, collate_fn=collote_fn)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    checkpoint = "bert-base-chinese"
    config = AutoConfig.from_pretrained(checkpoint)
    model = BertForPairwiseCLS.from_pretrained(checkpoint, config=config).to(device)
    
    learning_rate = 1e-5
    epoch_num = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=epoch_num * len(train_dataloader)
    )
    total_loss = 0
    best_acc = 0
    for i in range(epoch_num):
        print(f"Epoch {i + 1}/{epoch_num}\n-------------------------------")
        total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, i + 1, total_loss)
        valid_acc = test_loop(valid_dataloader, model, mode='Valid')
        if valid_acc > best_acc:
            best_acc = valid_acc
            print('saving new weights...\n')
            torch.save(model.state_dict(), f'model/finetune/epoch_{i + 1}_model_weights.bin')
    print("Done!")
    model.load_state_dict(torch.load('model/finetune/epoch_2_model_weights.bin'))
    test_loop(valid_dataloader, model, mode='Test')
