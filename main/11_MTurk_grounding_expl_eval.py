import pandas as pd
import os

in_dir = "data/MTurk/batches_output_grounding_comparison"
for fname in os.listdir(in_dir):
    df = pd.read_csv(in_dir+"/"+fname)
    machine_preferred, manual_preferred = 0, 0
    results, preferred = {}, {}
    for idx, data in df.iterrows():
        if data["Answer.howMuch1"] == 1 and data["Answer.howMuch2"] == 1 \
                and data["Answer.howMuch3"] == 1 and data["Answer.howMuch4"] == 1:
            continue
        for k_idx, (v, k) in enumerate(zip([data["Answer.howMuch1"],data["Answer.howMuch2"],data["Answer.howMuch3"],data["Answer.howMuch4"]],\
            data["Input.expl_order"].split("_"))):
            if k not in results:
                results[k] = []
                preferred[k] = 0
            results[k].append(v)
            if {"A":1,"B":2,"C":3,"D":4}[data["Answer.category.label"]]==k_idx:
                preferred[k] += 1
    for k, v in results.items():
        print(k, sum(v)/len(v), preferred[k])

