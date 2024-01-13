from tqdm.auto import tqdm
import torch
from torch import nn
from transformers import AdamW, get_scheduler
# tqdm库的作用是在循环程序执行中动态更新进度条，方便用户查看当前程序运行进度。

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
    learning_rate = 1e-5
    epoch_num = 3
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
            torch.save(model.state_dict(), f'epoch_{i + 1}_valid_acc_{(100 * valid_acc):0.1f}_model_weights.bin')
    print("Done!")
    model.load_state_dict(torch.load('epoch_3_valid_acc_74.1_model_weights.bin'))
    test_loop(valid_dataloader, model, mode='Test')