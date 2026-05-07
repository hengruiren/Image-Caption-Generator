import torch
import torch.nn as nn
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    AutoConfig,
    VisionEncoderDecoderModel,
    CLIPVisionModel,
    CLIPImageProcessor,
    GPT2LMHeadModel,
)


class MLPMapper(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=1024, out_dim=768, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.net(x)


class CaptioningModel(nn.Module):
    def __init__(self, encoder_name, decoder_name, use_mapper, tokenizer, device):
        super().__init__()
        self.use_clip = "clip" in encoder_name.lower()
        self.use_mapper = use_mapper
        self.device = device

        enc_config = AutoConfig.from_pretrained(encoder_name)
        dec_config = AutoConfig.from_pretrained(decoder_name)

        if self.use_clip:
            enc_dim = enc_config.vision_config.hidden_size
        else:
            enc_dim = enc_config.hidden_size

        dec_dim = dec_config.hidden_size

        dec_config.is_decoder = True
        dec_config.add_cross_attention = True

        if self.use_clip:
            encoder = CLIPVisionModel.from_pretrained(encoder_name)
            decoder = GPT2LMHeadModel.from_pretrained(decoder_name, config=dec_config)
            self.base = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
        else:
            self.base = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
                encoder_name, decoder_name, encoder_config=enc_config, decoder_config=dec_config
            )

        self.base.config.eos_token_id = tokenizer.eos_token_id
        self.base.config.pad_token_id = tokenizer.pad_token_id
        self.base.config.vocab_size = self.base.config.decoder.vocab_size
        self.base.config.decoder_start_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id

        for param in self.base.encoder.parameters():
            param.requires_grad = False

        if use_mapper:
            self.mapper = MLPMapper(in_dim=enc_dim, hidden_dim=1024, out_dim=dec_dim)
        else:
            self.mapper = None

    def forward(self, pixel_values, labels=None, encoder_outputs=None):
        if self.use_clip or self.mapper is not None:
            from transformers.modeling_outputs import BaseModelOutput
            with torch.no_grad():
                enc_out = self.base.encoder(pixel_values=pixel_values)
            hidden = enc_out.last_hidden_state
            if self.mapper is not None:
                hidden = self.mapper(hidden)
            return self.base(encoder_outputs=BaseModelOutput(last_hidden_state=hidden), labels=labels)

        return self.base(pixel_values=pixel_values, labels=labels)

    def generate(self, pixel_values, num_beams=4, max_length=40, **kwargs):
        if self.use_clip or self.mapper is not None:
            from transformers.modeling_outputs import BaseModelOutput
            with torch.no_grad():
                enc_out = self.base.encoder(pixel_values=pixel_values)
            hidden = enc_out.last_hidden_state
            if self.mapper is not None:
                hidden = self.mapper(hidden)
            return self.base.generate(encoder_outputs=BaseModelOutput(last_hidden_state=hidden), num_beams=num_beams, max_length=max_length, **kwargs)

        return self.base.generate(pixel_values=pixel_values, num_beams=num_beams, max_length=max_length, **kwargs)


def build_model(encoder_name, use_mapper, tokenizer, device):
    print(f"  Building model: encoder={encoder_name}  mapper={use_mapper}")
    model = CaptioningModel(encoder_name, "gpt2", use_mapper, tokenizer, device)
    return model.to(device)


def load_processors(encoder_name):
    if "clip" in encoder_name.lower():
        image_processor = CLIPImageProcessor.from_pretrained(encoder_name)
    else:
        image_processor = AutoImageProcessor.from_pretrained(encoder_name)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return image_processor, tokenizer


def parameter_count(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
