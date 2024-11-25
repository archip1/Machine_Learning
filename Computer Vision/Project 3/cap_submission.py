from functools import partial

import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from nltk.translate.bleu_score import corpus_bleu
from transformers import Seq2SeqTrainer
from transformers import default_data_collator
# ##########
# TODO: Add more imports
from transformers import VisionEncoderDecoderModel, GPT2Tokenizer, Seq2SeqTrainingArguments, VisionEncoderDecoderConfig,ViTImageProcessor,AutoModel, AutoModelForVision2Seq, get_linear_schedule_with_warmup, ViTConfig
from transformers import ViTImageProcessor, GPT2Tokenizer, VisionEncoderDecoderModel
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm

# ##########

class Args:
    """Configuration.
    """
    # Encoder-Decoder for captioning
    encoder = 'google/vit-base-patch16-224-in21k'
    decoder = 'gpt2'

    # Dataset path
    root_dir = "./flickr8k"

    # Save your model as "cap-vlm-{YOUR_CCID}"
    YOUR_CCID = "archi1"
    name = f"cap-vlm-{YOUR_CCID}"

    # Hyperparameters
    batch_size = 32
    lr = 5e-5
    epochs = 5

    # Generation cfgs
    # TODO: Add more as you see fit
    num_beams = 5
    max_length = 45     # TODO: Can play around

    # Train ops
    # TODO: Add more as you see fit
    logging_steps = 50

class FlickrDataset(Dataset):
    def __init__(
        self, 
        args, 
        processor, 
        tokenizer,
        mode: str = "train",
        ):
        assert mode in ["train", "val", "test"]
        self.args = args
        # ####################
        # TODO: Load Flickr8k dataset
        # TODO: Initialize vision encoder's processor
        # TODO: Initialize langauge decoder's tokenizer

        self.processor = processor

        self.tokenizer = tokenizer

        self.img_paths, self.captions = self.load_dataset(mode, args)
        # self.transform = transforms.Compose([
        #     transforms.Resize((224, 224)),  # Resize to the expected input size
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        # ])

        # ####################

    def load_dataset(self, mode, args):
        # Load image paths and captions from the Flickr8k dataset
        img_paths = []
        captions = []

        img_dir = os.path.join(args.root_dir, "images")
        
        # Example for loading (you need to implement the correct logic)
        if mode == "train":
            file = os.path.join(args.root_dir, "train.txt")
            with open(file) as f:
                for line in f:
                    img_path, caption = line.strip().split(';')
                    if img_path == "image":
                        continue
                    img_path = os.path.join(img_dir, img_path)
                    img_paths.append(img_path)
                    captions.append(caption)
        elif mode == "val":
            file = os.path.join(args.root_dir, "val.txt")
            with open(file) as f:
                for line in f:
                    img_path, caption = line.strip().split(';')
                    if img_path == "image":
                        continue
                    img_path = os.path.join(img_dir, img_path)
                    img_paths.append(img_path)
                    captions.append(caption)
        # Add test dataset loading as needed
        return img_paths, captions
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # ####################
        # TODO: Load and process image-caption data
        img = Image.open(self.img_paths[idx]).convert("RGB")
        # img = self.transform(img)
        # img = (img + 1) / 2
        # pixel_values = self.processor(img, return_tensors="pt").pixel_values.squeeze(0)
        pixel_values = self.processor(img, return_tensors="pt").pixel_values.contiguous().squeeze(0)

        # print("Image Shape:", pixel_values.shape)

        caption = self.captions[idx]
        # print("Caption:", caption)

        # inputs = self.tokenizer("<|beginoftext|> " + caption + " <|endoftext|>", padding='max_length', max_length=45, return_tensors="pt")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        caption = "<|beginoftext|>" + self.captions[idx] + "<|endoftext|>"         # NOTE: CHANGES

        # Ensure that padding is applied correctly with max_length
        tokenized_caption = self.tokenizer(
            caption, 
            padding="max_length",
            truncation=True,
            max_length=self.args.max_length,
            return_tensors="pt"
        )
        # print("Tokenized Caption Shape:", tokenized_caption["input_ids"].reshape(-1).shape)
        
        encoding = {
            "pixel_values": pixel_values,       # Return processed image as a tensor
            "labels": tokenized_caption["input_ids"].squeeze(),             # Return tokenized caption as a padded tensor
            "path": self.img_paths[idx],
            "captions": self.captions[idx],
        }
        # ####################

        return encoding

    
def train_cap_model(args):
    # Define your vision processor and language tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.decoder)
    processor = ViTImageProcessor.from_pretrained(args.encoder)

    tokenizer.add_special_tokens({
        # "pad_token" : "<|pad|>",
        "pad_token" : "[PAD]",
        "bos_token" : "<|beginoftext|>",
        "eos_token" : "<|endoftext|>"
    })

    vit_config = ViTConfig.from_pretrained(args.encoder)
    vit_config.num_attention_heads = 8

    # Define your Image Captioning model using Vision-Encoder-Decoder model
    # Reference: https://huggingface.co/docs/transformers/en/model_doc/vision-encoder-decoder
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(args.encoder, args.decoder, config = vit_config, trust_remote_code=True)    # NOTE: Send your model to GPU
    # model = AutoModel.from_pretrained(args.encoder, args.decoder, trust_remote_code=True)
    # model = AutoModelForVision2Seq.from_encoder_decoder_pretrained(args.encoder, args.decoder)

    model.decoder.resize_token_embeddings(len(tokenizer))       # NOTE: CHANGES

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id


    if hasattr(model.decoder.config, 'add_cross_attention'):
        model.decoder.config.add_cross_attention = True

    if torch.cuda.is_available():
        model = model.cuda()

    # Modify the embedding lookup table in decoder model and the tokenizer
    # to include bos_token "<|beginoftext|>" and pad_token "<|pad|>"
    # NOTE: The format of GPT2 inputs:
    # <|endoftext|> + article + " TL;DR: " + summary + <|endoftext|>
    # For captoning, we want:
    # <|beginoftext|> + caption + <|endoftext|>
    # followed by a number of paddings "<|pad|>"


    # Load train/val dataset
    train_dataset = FlickrDataset(args=args, mode ="train", tokenizer=tokenizer, processor=processor)
    val_dataset = FlickrDataset(args=args, mode ="val", tokenizer=tokenizer, processor=processor)
    # print("Train Dataset Length:", len(train_dataset))

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # for batch in train_loader:
    #     pixel_values = batch["pixel_values"]  # Shape: [32, C, H, W] (e.g., [32, 3, 224, 224])
    #     labels = batch["labels"]              # Shape: [32, max_length] (e.g., [32, 45])

    #     print("Pixel Values Shape:", pixel_values.shape)  # e.g., torch.Size([32, 3, 224, 224])
    #     print("Labels Shape:", labels.shape)              # e.g., torch.Size([32, 45])

    # Model configuration. 
    # Reference: https://huggingface.co/docs/transformers/en/model_doc/vision-encoder-decoder
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id       # NOTE: CHANGES

    # TODO: Play around with some generation config parameters
    # e.g. For beam search, you can potentially have a larger beam size of 5
    # Add more as you see fit
    model.generation_config.max_length = args.max_length #None
    model.generation_config.num_beams = args.num_beams #None

    # TODO: Define training arguments for Seq2Seq model (Seq2SeqTrainingArguments)
    # Reference: https://huggingface.co/docs/transformers/en/main_classes/trainer
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.name,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_steps=args.logging_steps,
        save_steps=args.logging_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        load_best_model_at_end=True,    
        report_to="none",
        predict_with_generate=True,     # NOTE: CHANGES
        # gradient_accumulation_steps=2,
        # optim="adamw_torch",
        # lr_scheduler_type="linear",
    )

    # Instantiate seq2seq model trainer
    compute_metrics = partial(compute_bleu_score, tokenizer=tokenizer)
    trainer = Seq2SeqTrainer(
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
    )

    # Start training
    # TODO: A good performing model should easily reach a BLEU score above 0.07
    # print("Shape of Image:", train_dataset[1]["pixel_values"].shape)
    trainer.train()
    trainer.save_model(args.name)

def load_trained_model(
    ckpt_dir: str,
    ):
    """TODO: Load your best trained model, processor and tokenizer.
    """
    # TODO: Load your model configuration
    # config = VisionEncoderDecoderConfig.from_pretrained(ckpt_dir)
    config = ViTConfig.from_pretrained(ckpt_dir)
    config.num_attention_heads = 8

    # TODO: Load encoder processor
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')           # NOTE: CHANGES

    # TODO: Load decoder tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(ckpt_dir)
    # tokenizer.add_special_tokens({"bos_token": "<|beginoftext|>", "pad_token": "[PAD]", "eos_token": "<|endoftext|>"})
    
    # TODO: Load your best trained model

    model = VisionEncoderDecoderModel.from_pretrained(ckpt_dir, trust_remote_code=True)           # NOTE: CHANGES
    # model = AutoModel.from_pretrained(ckpt_dir, trust_remote_code=True)
    # model = AutoModelForVision2Seq.from_encoder_decoder_pretrained(ckpt_dir)
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()               # NOTE: CHANGES

    return model, processor, tokenizer

def inference(
    img_path,
    model, 
    processor,
    tokenizer,
    ):
    """TODO: Example inference function to predict a caption for an image.
    """
    # TODO: Load and process the image
    image = Image.open(img_path).convert("RGB")
    img_tensor = processor(images=image, return_tensors="pt").pixel_values   # TODO: Preproces the image

    # Ensure your img_tensor is on GPU
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    # TODO: Generate the caption with VisionEncoderDecoderModel's generate API
    generated_ids = model.generate(img_tensor, max_length=45, num_beams=5)

    # generated_ids = model.generate(img_tensor, max_length=45, num_beams=5, temperature=0.7, top_k=50, top_p=0.95)

    # Tokens -> Str
    generated_caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return generated_caption

def compute_bleu_score(pred, tokenizer):
    """NOTE: DO NOT CHANGE.
    Compute BLEU score.
    NOTE: if you are interested in learning about the BLEU score, here are some interesting resources:
    https://www.geeksforgeeks.org/nlp-bleu-score-for-evaluating-neural-machine-translation-python/
    https://cloud.google.com/translate/automl/docs/evaluate#interpretation
    https://www.nltk.org/api/nltk.translate.bleu_score.html
    """

    pred_ids = pred.predictions
    labels_ids = pred.label_ids#.squeeze(1)

    # Decode predictions and labels while handling special tokens and padding
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == tokenizer.pad_token_id] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    # Prepare data for BLEU score calculation
    pred_bleu = [line.split() for line in pred_str]
    label_bleu = [[line.split()] for line in label_str]

    # Calculate BLEU score
    bleu_output = corpus_bleu(label_bleu, pred_bleu)
    bleu_score = round(bleu_output, 4)
    print("BLEU:", bleu_score)

    return {
        "bleu_score": bleu_score
    }