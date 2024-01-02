import os, csv, json, random
import pandas as pd

# norm, grounding, explanation
csv_data = [["path", "dial", "norm"]]

with open("data/prompt/norms_GPT3_proc.json","r") as f:
    GPT3norm = json.load(f)

fnames = os.listdir("data/prompt/norms_grounding/")
orig_fnames = ["data/prompt/norms_grounding/"+x for x in fnames if "_templateA" in x]
fnames = orig_fnames

df = pd.read_csv("data/prompt/norms_grounding_proc.csv")
fnamesA = [x for x in fnames if x.split("/")[-1].split(".json")[0] in df[df["Verdict_parsed"]==-1]["fID"].values]
fnamesB = [x for x in fnames if x.split("/")[-1].split(".json")[0] in df[df["Verdict_parsed"]==0]["fID"].values]
fnamesC = [x for x in fnames if x.split("/")[-1].split(".json")[0] in df[df["Verdict_parsed"]==1]["fID"].values]
fnamesA, fnamesC = random.sample(list(set(fnamesA)),k=33), random.sample(list(set(fnamesC)),k=34)
fnamesB = random.sample(list(set(fnames)-set(fnamesA)-set(fnamesC)),k=34-len(set(fnamesB))) + list(set(fnamesB))
fnames = fnamesA + fnamesC + fnamesB
swap_idx_start = len(fnamesA + fnamesC)

NormsKB = df["Norm"].values.tolist()

for fname_idx, fname in enumerate(fnames):
    with open(fname, "r") as f:
        prompted_stuff = json.load(f)
    try:
        dial = prompted_stuff[0]["prompt_txt"].split("\n\n")[0]
        dial = dial.split("Given the following dialogue:\n")[-1]
        norm = prompted_stuff[0]["prompt_txt"].split("given the social norm: ")[-1].split("\n\n")[0]
        if fname_idx >= swap_idx_start:
            norm = random.choice(list(set(NormsKB)-set(norm)))
        csv_data.append((fname, dial, norm))
    except:
        print(fname_idx)
print(len(set(fnamesA)),fnames[0],len(csv_data))

#with open("data/MTurk/batches_input_grounding/sample.csv", "w") as f:
#    json.dump(csv_data, f)
#for i in range(int(len(SocialChemCSV)/increment)):
#    with open("data/MTurk/batches_input/norm_evaluation_"+str(i)+".csv", "w") as f:
with open("data/MTurk/batches_input_grounding/sample_"+str(len(csv_data)-1)+".csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)
#    writer.writerows(header+SocialChemCSV[increment*i:increment*(i+1)])

