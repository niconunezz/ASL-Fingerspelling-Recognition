from types import SimpleNamespace
import torch
from transformers.models.speech_to_text import Speech2TextConfig
import augmentations as A

cfg = SimpleNamespace(**{})

# cfg.n_dim= 208
# cfg.n_heads = 4
# cfg.block_size = 384
# cfg.max_seq_len = 31
# cfg.total_lm = 130
#cfg.n_land_el = 4
# cfg.out_dim = 208

cfg.n_dim= 128
cfg.n_heads = 4
cfg.block_size = 64
cfg.max_seq_len = 20
cfg.total_lm = 42
cfg.n_land_el = 2
cfg.out_dim = 128

cfg.encoder_layers = 1
# cfg.vocab_size = 100277
cfg.vocab_size = 62

cfg.n_layer = 1
cfg.dropout= 0.3
cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
cfg.val_files = 16
cfg.epochs = 10
cfg.max_ex = None
cfg.aug = A.Compose([A.Resample(sample_rate = (0.3, 2.), p = 0.8),
                     #A.TemporalCrop(length = 384, p = 0.5),
                     A.TimeShift(p = 0.5),
                     A.SpatialAffine(scale=(0.7,1.3),shear=(-0.2,0.2),shift=(-0.15,0.15),degree=(-30,30),p=0.75)])

cfg.data_path = "data/ttensors"
cfg.csv_path = "data/sequences.csv"
cfg.folder = None

config = Speech2TextConfig.from_pretrained("facebook/s2t-small-librispeech-asr")
config.encoder_layers = 0
config.decoder_layers = 1
config.d_model = cfg.n_dim
# config.max_target_positions = 100277 #?
config.max_target_positions = 62
config.num_hidden_layers = 1
# config.vocab_size = 100277
config.vocab_size = 62
# config.bos_token_id = 100258
config.bos_token_id = 59
# config.eos_token_id = 100257
config.eos_token_id = 60
# config.decoder_start_token_id = 100258
config.decoder_start_token_id = 59
# config.pad_token_id = 100259
config.pad_token_id = 61
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