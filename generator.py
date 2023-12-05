import torch
from PIL import Image
from copy import deepcopy
from torchvision import transforms

from deephumor.data import tokenizers
from deephumor.imaging import memeify_image
from deephumor.experiments import text_to_seq, seq_to_text, split_caption
from deephumor.data.vocab import Vocab, build_vocab_from_file, SPECIAL_TOKENS

class MemeGenerator:
    def __init__(self, model, mode: str= 'word', device: str='cpu') -> None:
        self.mode = mode
        self.model = model
        self.device = device
        self.font_path = 'fonts/impact.ttf'
        if self.mode == 'word':
            self.vocab = Vocab.load('vocab/vocab_words.txt')
            self.tokenizer = tokenizers.WordPunctTokenizer()
        else:
            self.vocab = Vocab.load('vocab/vocab_chars.txt')
            self.tokenizer = tokenizers.CharTokenizer()
        
    def _proprocess_image(self, img: Image) -> torch.tensor:
        tfms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return tfms(img).unsqueeze(0).to(self.device)
    

    def _preprocessing_text(self, text: str) -> torch.tensor:
        tokens = self.tokenizer.tokenize(text)
        tokens += [SPECIAL_TOKENS['EOS']]
        tokens = [self.vocab.stoi[tok] for tok in tokens]
        return torch.tensor(tokens[:-1]).unsqueeze(0).to(self.device)
    
    def generate(
            self, 
            img_path: str, 
            caption: str=None, 
            labels: str=None, 
            T: int=1., 
            beam_size: int=7, 
            top_k: int=50, 
            delimiter: str='', 
            max_len: str=32
        ):

        img_pil = Image.open(img_path).convert('RGB')
        img_tensor = self._proprocess_image(deepcopy(img_pil))

        if caption is not None:
            caption = self._preprocessing_text(caption)

        if labels is None:
            with torch.no_grad():
                pred_seq = self.model.generate(image=img_tensor, 
                                               caption=caption, 
                                               temperature=T, 
                                               max_len=max_len, 
                                               beam_size=beam_size, 
                                               top_k=top_k)

        else: 
            with torch.no_grad():
                pred_seq = self.model.generate(image=img_tensor, 
                                               labels=labels,
                                               caption=caption, 
                                               temperature=T, 
                                               beam_size=beam_size, 
                                               top_k=top_k, 
                                               max_len=max_len)

        text = seq_to_text(pred_seq, vocab=self.vocab, delimiter=delimiter)
        top, bottom = split_caption(text, num_blocks=2)
        return memeify_image(img_pil, top, bottom, font_path=self.font_path)