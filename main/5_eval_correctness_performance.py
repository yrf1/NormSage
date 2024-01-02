import copy
import json
import math
import random
import pandas as pd
from sklearn.metrics import roc_auc_score


fname_intmd_correctness_check_results = "data/norm_discoveries/NormsKB_intmd_corr_check.json"

with open(fname_intmd_correctness_check_results, "r") as f:
    correctness_data = json.load(f)

df = pd.read_csv("data/eval_labels/Correctness_human_label_100_annotated.csv")
df1 = df[df["2"]=="Correct"][:50]
df2 = df[df["2"]=="Incorrect"][:50]
df = pd.concat([df1, df2])

TOK_LABEL_MAP = {"Correct":['yes'],"Incorrect":["no"]}

def prompt_correctness(norm):
    prompt_Q = "Consider this sentence:\n"+norm
    prompt_Q += '\n\nIs this a correct/acceptable social norm? Answer "yes" or "no", and then explain why.'
    prompt_result = openai.Completion.create(
        engine="text-davinci-003", prompt = prompt_Q,
        temperature=0.0, max_tokens=256, logprobs=5)
    correctness = prompt_result["choices"][0]["text"]
    token_logprobs = (prompt_result["choices"][0]["logprobs"]["tokens"], \
                            prompt_result["choices"][0]["logprobs"]["top_logprobs"])
    return correctness, token_logprobs

def prompt_correctness():
    for k, v in copy.deepcopy(correctness_data).items():
        source, norm = v["source"], v["norm"]
        if norm in df["1"].to_list():
            correctness, correctness_tokens_n_logprobs = prompt_correctness(norm)
            correctness_data[k]["correctness"] = correctness
            correctness_data[k]["correctness_tokens_n_logprobs"] = correctness_tokens_n_logprobs

    with open(fname_intmd_correctness_check_results, "w") as f:
        json.dump(correctness_data, f)

def conf_score_helper(label, pred_label, tokens, logprobs):
    max_match_prob = -99
    token_idx, k = [(x_idx, x) for x_idx, x in enumerate(tokens) if (x.lower()=="yes" or x.lower()=="no")][0]
    match_prob = logprobs[token_idx]
    try:
        match_prob = max([v for k, v in match_prob.items() if k.lower()==TOK_LABEL_MAP[pred_label][0]])
    except:
        match_prob = [v for k, v in match_prob.items() if k.lower()=="maybe"]
        match_prob = match_prob[0] if len(match_prob)>0 else 0.001
    if match_prob>max_match_prob:
        max_match_prob = match_prob
    max_alt_ans_prob = -99
    token_idx, k = [(x_idx, x) for x_idx, x in enumerate(tokens) if (x.lower()=="no" or x.lower()=="yes")][0]
    alt_ans_prob = logprobs[token_idx]
    alt_pred_label = [x for x in list(TOK_LABEL_MAP.keys()) if x!=pred_label][0]
    try:
        alt_ans_prob = max([v for k, v in alt_ans_prob.items() if k.lower()==TOK_LABEL_MAP[alt_pred_label][0]])
    except:
        alt_ans_prob = [v for k, v in alt_ans_prob.items() if k.lower()=="maybe"]
        alt_ans_prob = alt_ans_prob[0] if len(alt_ans_prob)>0 else 0.001
    if alt_ans_prob>max_alt_ans_prob:
        max_alt_ans_prob = alt_ans_prob
    match_prob, alt_ans_prob = math.exp(max_match_prob), math.exp(max_alt_ans_prob)
    return match_prob/(match_prob+alt_ans_prob)

def compute_stats():
    C, W, auc = 0, 0, []
    for idx, data in df.iterrows():
        if "Yes, " in data["3"] and "Correct"==data["2"]:
            C += 1
        elif "No, " in data["3"] and "Incorrect"==data["2"]:
            C += 1
        elif data["2"] in ["Correct","Incorrect"]:
            W += 1
        this_correctness_data = [v for (k,v) in correctness_data.items() \
                if v["source"]==data["0"] and v["norm"]==data["1"]][0]
        if "maybe" in this_correctness_data["correctness"].lower():
            continue
        elif "yes " not in this_correctness_data["correctness"].lower().replace(",","") and \
                "no " not in this_correctness_data["correctness"].lower().replace(",",""):
            continue
        tokens = this_correctness_data["correctness_tokens_n_logprobs"][0]
        logprobs = this_correctness_data["correctness_tokens_n_logprobs"][1]
        prob = conf_score_helper(data["2"],data["2"], tokens, logprobs)
        auc.extend([(1, prob),(0, 1-prob)])
        
    print(C, W) #46 15
    auc = roc_auc_score([x[0] for x in auc], [x[1] for x in auc])
    print(C/(C+W), auc)

if __name__ == "__main__":
    compute_stats()


