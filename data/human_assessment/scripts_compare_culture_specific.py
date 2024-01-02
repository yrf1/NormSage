from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json, pandas as pd, numpy as np
from nltk.tokenize import sent_tokenize
from datetime import datetime
import torch

from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
punkt_param = PunktParameters()
punkt_param.abbrev_types = set(['e.g', 'dr', 'vs', 'mr', 'mrs', 'prof', 'inc'])
sentence_splitter = PunktSentenceTokenizer(punkt_param)

device = "cuda"

model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli").to(device)
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")

with open("data/norm_discoveries/NormSage_wIndc.json", "r") as f:
    norms = json.load(f)

norms_per_culture = {}
norm_pairs_per_culture = {}

cultures = ["Brit","Pakistan", "India", "Taiwanese", "American", "White", "African-American"]

for k, v in norms.items():
    for norm in v["norms"]:
        if ":" == norm[-1]:
            continue
        try:
            if ", while " in norm or ", whereas " in norm or (", but " in norm and len([x for x in cultures if x.lower() in norm.lower()])>1) or len(sentence_splitter.tokenize(norm))>1:            
                kw = ", while " if ", while " in norm \
                    else ", whereas " if ", whereas " in norm else ", but " if ", but " in norm else None
                for norm_ in (norm.split(kw) if kw!=None else sentence_splitter.tokenize(norm)):
                    culture = [x for x in cultures if x.lower() in norm_.lower()][0]
                    if culture not in norms_per_culture:
                        norms_per_culture[culture] = []
                    norms_per_culture[culture].append(norm_)
            else:
                if " culture, " in norm:
                    culture = norm.split(" culture, ")[0]
                    culture = culture.replace("In ","")
                    norm = norm[norm.find(" culture, ")+len(" culture, "):]
                    if culture not in norms_per_culture:
                        norms_per_culture[culture] = []
                    norms_per_culture[culture].append(norm)
        except:
            pass
for (culture1, culture2) in [("Brit","Pakistan"),("India","American"),("Taiwanese", "American"),("White", "African-American")]: 
    if (culture1, culture2) not in norm_pairs_per_culture:
        norm_pairs_per_culture[(culture1, culture2)] = []
    for norm1 in norms_per_culture[culture1]:
        print(datetime.now())
        for norm2 in norms_per_culture[culture2]:
            if norm1 != norm2:
                x = tokenizer.encode(norm1, norm2, return_tensors='pt',
                     truncation_strategy='only_first')
                logits = model(x.to(device))[0]
                prediction = logits.argmax(1).detach().item()
                if prediction == 2:
                    if (norm2,norm1) not in norm_pairs_per_culture[(culture1, culture2)] and \
                            (norm1, norm2) not in norm_pairs_per_culture[(culture1, culture2)]:
                        norm_pairs_per_culture[(culture1, culture2)].append((norm1, norm2))
        if len(norm_pairs_per_culture[(culture1, culture2)])>20:
            break

for culture, norm_pairs in norm_pairs_per_culture.items():
    print(culture, len(norm_pairs))

with open("data/human_assessment/mturk_culture_comparison.json", "w") as f:
    json.dump(norm_pairs_per_culture, f, indent=4)
