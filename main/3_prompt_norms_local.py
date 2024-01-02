"""
Data Filepaths:
data/norm_discoveries/NormSage_base.json 


"""
import json
import torch
import random
import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BartForConditionalGeneration #AutoModelForSeq2SeqLM #GPT2ForSeq2SeqLM #SequenceClassification, AutoModelForSeq2SeqLM


parser = argparse.ArgumentParser()
parser.add_argument('--mode')  
args = parser.parse_args()

mode = args.mode

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "facebook/bart-large" #openai-gpt" #microsoft/DialogRPT-updown"

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        fname = "data/norm_discoveries/NormSage_base.json"
        with open(fname, "r") as f:
            self.data = json.load(f)
        self.fnames = list(self.data.keys())
        self.fnames = [x for x in self.fnames if len(self.data[x]["norms"])>0]
        with open("data/human_assessment/fID_list_100.txt", "r") as f:
            fnames_test = f.read().split("\n")
            fnames_test = [x+".json" for x in fnames_test]
        if mode=="train":
            self.fnames = [x for x in self.fnames if x not in fnames_test]
        if mode=="test":
            self.fnames = fnames_test
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_name == "openai-gpt":
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        k = self.fnames[idx]
        norms = self.data[k]["norms"]
        norms = random.choice(norms) #,k=1)
        with open("data/dialchunk/"+k, "r") as f:
            txt = json.load(f)["txt"]
        #print(self.tokenizer(txt,return_tensors="pt")["input_ids"].size())
        txt = self.tokenizer(txt, max_length=64, pad_to_max_length=True, return_tensors="pt")
        norms = self.tokenizer(norms, max_length=64, pad_to_max_length=True, return_tensors="pt")
        txt['input_ids'] = txt['input_ids'].squeeze(0).to(device)
        txt['attention_mask'] = txt['attention_mask'].squeeze(0).to(device)
        return txt, norms['input_ids'].squeeze(0).to(device)

dataset = CustomDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=36, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

def shift_tokens_right(input_ids, pad_token_id=1, decoder_start_token_id=2):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids

if mode=="train":
    loss_tracker_by_ep = []
    for ep in range(10):
        loss_tracker = []
        for idx, (x,y) in enumerate(dataloader):
            optim.zero_grad()
            x["decoder_input_ids"] = shift_tokens_right(y)
            logits = model(**x).logits
            loss = loss_fn(torch.swapaxes(logits,2,1), y)
            loss.backward()
            optim.step()
            loss_tracker.append(loss.item())
            if idx%300==0:
                print("~~~~~~~")
                x["input_ids"] = x["input_ids"][20,:].unsqueeze(0)
                x["attention_mask"] = x["attention_mask"][20,:].unsqueeze(0)
                print(x["input_ids"].size(), x["attention_mask"].size())
                out = model.generate(x["input_ids"], attention_mask=x["attention_mask"], max_length=64+1, num_beams=4)
                print(dataset.tokenizer.decode(x["input_ids"][0],skip_special_tokens=True))
                print("~", sum(loss_tracker)/len(loss_tracker))
                print(dataset.tokenizer.decode(out[0],skip_special_tokens=True))
        loss_tracker_by_ep.append(sum(loss_tracker)/len(loss_tracker))
        print(loss_tracker_by_ep)

    torch.save(model.state_dict(), 'ckpts/local_normsage_mini.pth')

if mode=="test":
    print("Loading model checkpoint...")
    model.load_state_dict(torch.load('ckpts/local_normsage_mini.pth'))
    print("Fined loading model checkpoint")
    results = {}
    for idx, (x,y) in enumerate(test_dataloader):
        out = model.generate(**x) #, beam=4)
        dial = dataset.tokenizer.decode(x["input_ids"][0],skip_special_tokens=True)
        out = dataset.tokenizer.decode(out[0],skip_special_tokens=True)
        print(dial)
        print(out)
        fname = test_dataset.fnames[idx]
        results[fname] = (dial, out)
    with open("data/norm_discoveries/NormSage_mini.json", "w") as f:
        json.dump(results, f)
