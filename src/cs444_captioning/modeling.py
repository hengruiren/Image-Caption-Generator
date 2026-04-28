from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel


def load_processors(cfg):
    image_processor = AutoImageProcessor.from_pretrained(cfg.encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.decoder_name)
    tokenizer.pad_token = tokenizer.eos_token
    return image_processor, tokenizer


def build_vit_gpt2_baseline(cfg, tokenizer, device, freeze_encoder=True):
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        cfg.encoder_name,
        cfg.decoder_name,
    )
    model.config.decoder_start_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.max_length = cfg.max_length
    model.config.num_beams = cfg.generation_num_beams
    model.config.decoder.is_decoder = True
    model.config.decoder.add_cross_attention = True

    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    return model.to(device)


def parameter_count(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable

