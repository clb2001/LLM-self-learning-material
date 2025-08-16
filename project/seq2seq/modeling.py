from torch import nn
from transformers.models.marian import MarianPreTrainedModel, MarianModel, MarianMTModel
import torch
import logging
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    Seq2SeqLMOutput,
)


class MarianForMT(MarianMTModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = MarianMTModel(config)
        target_vocab_size = config.decoder_vocab_size
        # 这行代码的意思是在PyTorch模型中注册一个名为"final_logits_bias"的缓冲区，并初始化为一个全零的张量，维度为(1, target_vocab_size)。
        # 这个缓冲区可以被模型访问和使用，通常用于存储模型的参数或其他需要持久化的数据。
        self.register_buffer("final_logits_bias", torch.zeros((1, target_vocab_size)))
        self.lm_head = nn.Linear(config.d_model, target_vocab_size, bias=False)
        self.post_init() # 这个方法在对象初始化完成后自动调用，可以用来执行一些需要在对象创建后立即执行的操作。
        
    def forward(self, x):
        outputs = self.model(**x)
        sequence_output = outputs.decoder_hidden_states
        lm_logits = self.lm_head(sequence_output) + self.final_logits_bias
        
        return_dict = None
        labels = x['labels']
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.decoder_vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
    
    def other_func(self):
        pass
