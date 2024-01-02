import os, json, torch
from transformers import AutoModel, AutoTokenizer


base_dir = "data/prompt/norms_GPT3_new/"

with open(base_dir+"prompted_results_all.json", "r") as f:
    data = json.loads(f.read())

boiler_templates = ['Some cultural norms that may be related to this situation include:', \
                    'Some cultural norms related to this situation could be:']

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").eval()
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

ROTs = {}
for fname, v in data.items():
    dial, Qs, norms = v["dial"], v["Qs"], v["norms"]
    speakers = list(set([x.split(": ")[0] for x in dial.split("\n") if len(x)>0]))
    Q_map, q_counter, Q_set = {}, 0, list(set([x[0] for x in Qs]))
    for Q, q_idx in Qs:
        for i in range(q_counter, q_counter+q_idx):
            Q_map[i] = Q_set.index(Q)
        q_counter += q_idx
    norm_pairs, new_norms, skip_idx = [], [], []
    # Save computation time
    norm_idx2embed_cache = {}
    for i, norm in enumerate(norms):
        tok = tokenizer(norm.replace("It's","It is"),return_tensors="pt")
        embed = model(**tok).pooler_output
        norm_idx2embed_cache[i] = embed
    for i1, norm1 in enumerate(norms):
        if i1 not in skip_idx and norm1 not in boiler_templates:
            if norm1[-16:] != " are as follows:" and not any(s in norm1 for s in speakers):
                norm1 = norm1[3:] if norm1[1:3]==". " else norm1
                new_norms.append((Q_map[i1], norm1))
            skip_idx.append(i1)
        for i2, norm2 in enumerate(norms):
            if i2 > i1:
                norm_pairs.append((i1, i2, norm1, norm2))
                embed1, embed2 = norm_idx2embed_cache[i1], norm_idx2embed_cache[i2]
                cos_score = cos(embed1, embed2).item()
                if cos_score <= 0.998 and i2 not in skip_idx \
                        and norm2 not in [x[1] for x in new_norms] \
                        and norm2 not in boiler_templates:
                    if norm2[-16:] != " are as follows:" and not any(s in norm2 for s in speakers):
                        norm2 = norm2[3:] if norm2[1:3]==". " else norm2
                        new_norms.append((Q_map[i2],norm2))
                if cos_score > 0.998 and (Q_map[i2],norm2) in new_norms:
                    idx = new_norms.index((Q_map[i2],norm2))
                    del new_norms[idx]
                skip_idx.append(i2)
    ROTs[fname] = {"dial":dial, "Qs":Q_set, "norms": new_norms}

with open(base_dir+"prompted_results_all_proc.json", "w") as f:
    json.dump(ROTs, f, indent=4)
