from model import *
import torch
from torchtext.datasets import UDPOS
from torchtext.vocab import build_vocab_from_iterator
from transformers import BertModel,BertTokenizer
import gdown

TRANSFORMER = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(TRANSFORMER) # tokenizer for BERT
init_token = tokenizer.cls_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token
sep_token = tokenizer.sep_token
init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)
sep_token_idx = tokenizer.convert_tokens_to_ids(sep_token)
max_input_length = tokenizer.max_model_input_sizes[TRANSFORMER]



train_datapipe = UDPOS(split="train")
valid_datapipe = UDPOS(split="valid")
pos_vocab = build_vocab_from_iterator(
    [i[1] for i in list(train_datapipe)],
    specials=[init_token, pad_token, sep_token],
)
T_CAL = torch.tensor([i for i in range(pos_vocab.__len__())])

bert = BertModel.from_pretrained(TRANSFORMER)
crf = NeuralCRF(
    pad_idx_word=pad_token_idx,
    pad_idx_pos=pos_vocab[pad_token],
    bos_idx=init_token_idx,
    eos_idx=sep_token_idx,
    bot_idx=pos_vocab[init_token],
    eot_idx=pos_vocab[sep_token],
    t_cal=T_CAL,
    transformer=bert,
    beta=1)

#download model from google drive
file_id = '1dy3cX-4xSL00rx0ekq1VvMZw67pRKlQr'
output = 'pos_model.pt'
url = f'https://drive.google.com/uc?id={file_id}'
gdown.download(url, output, quiet=False)

print(f"Downloaded '{output}' from Google Drive")

PATH = "pos_model.pt"
crf.load_state_dict(torch.load(PATH))
crf.eval()
sentence = input("Enter a sentence: ")
sentence_tok = tokenizer(str(sentence),return_tensors="pt")
input_ids = sentence_tok["input_ids"]
pos = crf(input_ids)
res = [pos_vocab.get_itos()[i.int()] for i in pos[0]]
sentence = sentence.split()
d = {sentence[i]: res[i] for i in range(len(sentence))}
print(str(d)[1:-1])
