from torch import nn
from transformers.models.marian import MarianPreTrainedModel, MarianModel
import torch

class MarianForMT(MarianPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = MarianModel(config=config)
        target_vocab_size = config.decoder_vocab_size
        # 这行代码的意思是在PyTorch模型中注册一个名为"final_logits_bias"的缓冲区，并初始化为一个全零的张量，维度为(1, target_vocab_size)。
        # 这个缓冲区可以被模型访问和使用，通常用于存储模型的参数或其他需要持久化的数据。
        self.register_buffer("final_logits_bias", torch.zeros((1, target_vocab_size)))
        self.lm_head = nn.Linear(config.d_model, target_vocab_size, bias=False)
        self.post_init() # 这个方法在对象初始化完成后自动调用，可以用来执行一些需要在对象创建后立即执行的操作。
        
    def forward(self, x):
        output = self.model(**x)
        sequence_output = output.last_hidden_state
        lm_logits = self.lm_head(sequence_output) + self.final_logits_bias
        return lm_logits
    
    def other_func(self):
        pass
