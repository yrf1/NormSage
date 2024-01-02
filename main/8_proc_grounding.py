import json, csv, os


result = [("fID","Dialogue","Norm","Full_Grounding_Prompted", \
        "Verdict_parsed", "Explanation1_parsed","Explanation2_parsed")]
uh = []
dir_prompted_grounding = "data/prompt/norms_grounding"
for fname in os.listdir(dir_prompted_grounding):
    if "_templateA" not in fname:
        continue
    with open(dir_prompted_grounding+"/"+fname, "r") as f:
        data = json.load(f)
    fID = fname.split(".json")[0]
    for dial_norm_pair in data:
        inp_prompt = dial_norm_pair["prompt_txt"]
        dial = inp_prompt.split("\n\n")[0]
        norm = inp_prompt.split("\n\n")[1].split("And given the social norm: ")[1]
        out_grounding = dial_norm_pair["grounding_result"]
        out_grounding = out_grounding["choices"][0]["text"].lstrip("\n")
        verdict = 0 if "does not apply to the given dialogue scenario" in out_grounding \
                else -1 if "is a violation" in out_grounding else 1
        expl_gen1 = out_grounding.split("\n")[0].split("because ")[-1]
        try:
            expl_gen2 = out_grounding.split(" violat")[1].split("because ")[-1]
        except:
            expl_gen2 = ""
        result.append((fID, dial, norm, out_grounding, verdict, expl_gen1, expl_gen2))
        if verdict == 1:
            uh.append(result[-1])

print(len(result), len([x for x in result if x[-3]==1]), \
        len([x for x in result if x[-3]==0]), len([x for x in result if x[-3]==-1]))

with open("data/prompt/norms_grounding_proc.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(result)

