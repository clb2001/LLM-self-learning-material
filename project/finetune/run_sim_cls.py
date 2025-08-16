from tqdm.auto import tqdm
import torch
# tqdm库的作用是在循环程序执行中动态更新进度条，方便用户查看当前程序运行进度。

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
