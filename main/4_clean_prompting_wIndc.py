import os, json, torch, spacy
from transformers import AutoModel, AutoTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters


base_dir = "data/prompt/norms_GPT3_wIndc/"

with open(base_dir+"prompted_results_all.json", "r") as f:
    data = json.loads(f.read())

boiler_templates = ['Some cultural norms that may be related to this situation include:', \
                    'Some cultural norms related to this situation could be:']

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").eval()
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

punkt_param = PunktParameters()
punkt_param.abbrev_types = set(['o.j','gov', 'sen', 'dr', 'vs', 'mr', \
                               'mrs', 'prof', 'inc', 'ms' 'ph.d', 'i.e', 'e.g'])
sentence_splitter = PunktSentenceTokenizer(punkt_param)

nlp = spacy.load("en_core_web_sm")

ROTs = {}
for fname, v in data.items():
    dial, Qs, norms = v["dial"], v["Qs"], v["norms"]
    speakers = list(set([x.split(": ")[0] for x in dial.split("\n") if len(x)>0]))
    cultures = list(set([n.split(" ")[-1] for n in Qs]))
    new_norms, Q_set = [], []
    for n_idx, n in enumerate(norms):
        if n[1:3] == ". ":
            n = n[3:]
        if n[-len(" include:"):] == " include:" or n[-5:]==" are:":
            continue
        sent_split, s_skip = sentence_splitter.tokenize(n), []
        for s_idx, s in enumerate(sent_split):
            if s_idx in s_skip:
                continue
            if s_idx < len(sent_split)-1:
                if "this " in sent_split[s_idx+1].lower():
                    s = s+" "+sent_split[s_idx+1]
                    s_skip.append(s_idx+1)
            s = s.lstrip()
            if len(s.split(" "))<5:
                continue
            if not any(speaker in s for speaker in speakers):
                s = s[5:] if s[:5]=="that " else s
                if not any(c[:5] in s for c in cultures):
                    culture = Qs[n_idx].split(" ")[-1]
                    if s[0].upper():
                        s = s[0].lower() + s[1:]
                    s = "In "+culture+" culture, " + s
                if " also " in s and ", " not in s:
                    s = s.replace(" also ", "")
                if any(t.dep_ == "nsubj" for t in nlp(s)) and \
                        any(t.pos_ == "VERB" for t in nlp(s)): 
                    new_norms.append(s)
                Q_set.append(Qs[n_idx][0])
    # TODO: do another round of filtering
    print(len(new_norms))
    ROTs[fname] = {"dial":dial, "Qs":Q_set, "norms": new_norms}

with open(base_dir+"prompted_results_all_proc.json", "w") as f:
    json.dump(ROTs, f, indent=4)

