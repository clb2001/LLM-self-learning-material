# 对于分类任务，可以直接使用我们前面介绍过的 AutoModelForSequenceClassification 类来完成。
# 但是在实际操作中，除了使用预训练模型编码文本外，我们通常还会进行许多自定义操作，因此在大部分情况下我们都需要自己编写模型。
import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel
from transformers import AutoModel, AutoConfig

# #  这种方法相当于在Transformers模型外又包了一层，因此无法再调用Transformers库预置的模型函数
# class BertForPairwiseCLS(nn.Module):
#     def __init__(self):
#         super(BertForPairwiseCLS, self).__init__()
#         self.bert_encoder = AutoModel.from_pretrained(checkpoint)
#         self.dropout = nn.Dropout(0.1)
#         self.classifier = nn.Linear(768, 2)
        
#     def forward(self, x):
#         bert_output = self.bert_encoder(**x)
#         cls_vectors = bert_output.last_hidden_state[:, 0, :]
#         cls_vectors = self.dropout(cls_vectors)
#         logits = self.classifier(cls_vectors)
#         return logits

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
    

if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    checkpoint = "bert-base-chinese"
    # 原来的调用方法
    # model = BertForPairwiseCLS().to(device)
    config = AutoConfig.from_pretrained(checkpoint)
    model = BertForPairwiseCLS.from_pretrained(checkpoint, config=config).to(device)
    print(model)
