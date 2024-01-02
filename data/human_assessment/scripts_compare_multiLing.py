"""
Target Data
fID, dial, norm, relevance, well-formedness, correctness, insightfulness, relatableness

Shared Drive location:
https://uofi.box.com/s/7iff23ja45ir62jdajjwubqf4w27egzd
"""
import pandas as pd
import random
import json
import os

"""
with open("data/norm_discoveries/NormSage_wSchema_new.json", "r") as f:
    data = json.load(f)

english_data, chinese_data = [], []
print(set([x.split("_")[0] for x in list(data.keys())]))
for k, v in data.items():
    dial = v["dial"]
    norms = v["norms"]
    norms = [x[-1] for x in norms] if len(norms[0])==2 else norms
    norms = random.sample(norms, min(3,len(norms)))
    if len(english_data)<50 and k[:2]!="M0":
        for norm in norms:
            english_data.append((k,dial,norm))
    if len(chinese_data)<50 and k[:2]=="M0":
        for norm in norms:
            chinese_data.append((k,dial,norm))
tot_data = english_data + chinese_data
df = pd.DataFrame(tot_data, columns=["fID", "dial", "norm"])
df["rlv"] = ""
df["well-formed"] = ""
df["cor"] = ""
df["insight"] = ""
df["relatable"] = ""
df = df.sample(frac=1)
df.to_csv("data/human_assessment/NormComparisons/mutliLing_norm_comparison.csv", index=False, encoding='utf_8_sig')
"""
df = pd.read_csv("data/human_assessment/NormComparisons/mutliLing_norm_comparison_annotated.csv", encoding='utf_8_sig')
df = df[~df["cor"].isna()]
print(df.shape, df.columns)
print(df["rlv"].mean(), df["well-formed"].mean(), df["cor"].mean(), \
        df["insight"].mean(), df["relatable"].mean())


