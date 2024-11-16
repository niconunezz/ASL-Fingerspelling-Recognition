from types import SimpleNamespace
import torch
from transformers.models.speech_to_text import Speech2TextConfig
import augmentations as A

cfg = SimpleNamespace(**{})

cfg.n_dim= 208
cfg.n_heads = 4
cfg.block_size = 384
cfg.max_seq_len = 31
cfg.encoder_layers = 8
cfg.vocab_size = 100277
cfg.n_layer = 8
cfg.dropout= 0.1
cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
cfg.val_files = 16
cfg.epochs = 10
cfg.max_ex = None
cfg.aug = A.Compose([A.Resample(sample_rate = (0.3, 2.), p = 0.8),
                     #A.TemporalCrop(length = 384, p = 0.5),
                     A.TimeShift(p = 0.5),
                     A.SpatialAffine(scale=(0.7,1.3),shear=(-0.2,0.2),shift=(-0.15,0.15),degree=(-30,30),p=0.75)])

config = Speech2TextConfig.from_pretrained("facebook/s2t-small-librispeech-asr")
config.encoder_layers = 0
config.decoder_layers = 4
config.d_model = cfg.n_dim
config.max_target_positions = 100277 #?
config.num_hidden_layers = 1
config.vocab_size = 100277
config.bos_token_id = 100258
config.eos_token_id = 100257
config.decoder_start_token_id = 100258
config.pad_token_id = 100259
config.num_conv_layers = 0
config.conv_kernel_sizes = []
config.max_length = cfg.n_dim
config.input_feat_per_channel = cfg.n_dim
config.num_beams = 1
config.attention_dropout = 0.2
    # config.dropout = 0.2
config.decoder_ffn_dim = 512
config.init_std = 0.02

cfg.decoder_cf = config