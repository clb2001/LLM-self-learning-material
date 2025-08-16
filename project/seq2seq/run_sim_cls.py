from tqdm.auto import tqdm
import torch
import numpy as np
from sacrebleu.metrics import BLEU

bleu = BLEU()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_input_length = 128
max_target_length = 128

def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch - 1) * len(dataloader)
    model.train()
    for batch, batch_data in enumerate(dataloader, start=1):
        batch_data = batch_data.to(device)
        outputs = model(batch_data)
        loss = outputs.loss
        # 实际上的计算方法
        # loss_fct = CrossEntropyLoss()
        # loss = loss_fct(lm_logits.view(-1, self.config.decoder_vocab_size), labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss

def test_loop(tokenizer, dataloader, model):
    preds, labels = [], []
    model.eval()
    for batch_data in tqdm(dataloader):
        batch_data = batch_data.to(device)
        with torch.no_grad():
            genereted_tokens = model.generate(
                batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_length=max_target_length
            ).cpu().numpy()
        decoded_preds = tokenizer.batch_decode(genereted_tokens, skip_special_tokens=True)
        label_tokens = np.where(label_tokens != 100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)
        preds += [pred.strip() for pred in decoded_preds]
        labels += [label.strip() for label in decoded_labels]
    bleu_score = bleu.corpus_score(preds, labels).score
    print(f"BLEU: {bleu_score:>0.2f}\n")
    return bleu_score
