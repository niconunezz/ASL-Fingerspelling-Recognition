import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from transformers.models.speech_to_text.modeling_speech_to_text import shift_tokens_right, Speech2TextDecoder


class Decoder(nn.Module):
    def __init__(self, decoder_config):
        super(Decoder, self).__init__()
        
        self.config = decoder_config
        self.decoder = Speech2TextDecoder(decoder_config) 
        self.lm_head = nn.Linear(decoder_config.d_model, decoder_config.vocab_size, bias=False)
        
        self.decoder_start_token_id = decoder_config.decoder_start_token_id
        self.decoder_pad_token_id = decoder_config.pad_token_id #used for early stopping
        self.decoder_end_token_id= decoder_config.eos_token_id
        
    def forward(self,x, labels=None, attention_mask = None, encoder_attention_mask = None):
        
        if labels is not None:
            decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
            
        decoder_outputs = self.decoder(input_ids=decoder_input_ids,
                                       encoder_hidden_states=x, 
                                       attention_mask = attention_mask,
                                       encoder_attention_mask = encoder_attention_mask)
        lm_logits = self.lm_head(decoder_outputs.last_hidden_state)
        return lm_logits