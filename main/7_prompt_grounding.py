"""
Speaker-Level Norm Instantiation Grounding Results
"""
import os, json, pandas as pd
import openai
import math
import copy
from sklearn.metrics import roc_auc_score


openai.organization = os.getenv("OPENAI_ORG")
openai.api_key = os.getenv("OPENAI_API_KEY")

def init_sample_n_annotate():
    with open("data/norm_discoveries/NormSage_wSchema.json","r") as f:
        GPT3norm = json.load(f)

def prompt(results_fname):
    df = pd.read_csv("data/eval_labels/GroundingAnnotationSpeakerLevel.csv")
    print(df[df["grounding1"].isna()==False].shape, df[df["grounding2"].isna()==False].shape, df[df["grounding3"].isna()==False].shape)
    responses = {}
    if os.path.exists(results_fname):
        with open(results_fname, "r") as f:
            responses = json.load(f)
    for fname_idx, data in df.iterrows():
        fname = data["fname"].replace(".txt",".json")
        if fname not in responses:
            responses[fname] = {}
        with open("data/dialchunk/"+fname,"r") as f:
            dial = json.load(f)["txt"].replace("</i>","").replace("<i>","")
        for label_idx, label in zip([1,2,3],["violate","entail","neutral"]):
            if data["grounding"+str(label_idx)] != data["grounding"+str(label_idx)]:
                continue

            speaker_ln = data["grounding"+str(label_idx)].split("Norm")[0].split("(Dialogue:")[-1]
            speaker = speaker_ln.split(":")[0].lstrip()
            norm = data["grounding"+str(label_idx)].split("Norm: ")[-1].split(")")[0]
            if label in responses[fname]:
                if norm != responses[fname][label]["norm"]:
                    fname = fname.replace(".json", "_dup.json")
                    if fname not in responses:
                        responses[fname] = {}
            if label in responses[fname]:
                continue
            prompt_txt = "Given the following dialogue:\n"
            prompt_txt += dial
            prompt_txt += "\n\nAnd given the social norm: "+norm+"\n\n"
            if ":" in speaker_ln and "2" not in results_fname:
                prompt_txt += "Explain whether what's spoken by "+str(speaker)+" in the conversation " + \
                          "<entails, is irrelevant with, or contradicts> the social norm and why."
            else:
                prompt_txt += "Explain whether the conversation " + \
                          "<entails, is irrelevant with, or contradicts> the social norm and why."
            response = openai.Completion.create(
                  engine="text-davinci-003",
                  prompt = prompt_txt,
                  temperature=0,
                  logprobs=16,
                  max_tokens=80,
                  top_p=1
            )
            responses[fname][label] = {"norm":norm, "prompt_txt":prompt_txt, \
                        "grounding_result":response["choices"][0]["text"].lstrip(), \
                        "tokens":response["choices"][0]["logprobs"]["tokens"], \
                        "logprobs":response["choices"][0]["logprobs"]["top_logprobs"]}
        with open(results_fname, "w") as f:
            json.dump(responses, f)
    
def eval(results_fname):
    with open(results_fname, "r") as f:
        NormSage = json.load(f)
    c, w, correct, wrong, two_cls_probs, probs = 0, 0, 0, 0, [], []
    label_tok_map = {"violate":["contradict","cont"],"entail":["entail","ent"],"neutral":["irrelevant","irre"]}
    for fname, v in NormSage.items():
        for label_idx, label in zip([1,2,3],["violate","entail","neutral"]):
            if label not in NormSage[fname]:
                continue
            response = NormSage[fname][label]
            result = NormSage[fname][label]["grounding_result"]
            tokens = NormSage[fname][label]["tokens"]
            if (label == "violate" or label == "entail") and (("contradict" in result and "not contradict" not in result) \
                    or ("entail" in result and "not entail" not in result)):
                c += 1
            elif (label == "neutral" and ("irrelevant" in result or "does not entail" in result \
                    or "does not violate" in result or "does not contradict" in result)):
                c += 1
            else:
                w += 1
            if label == "violate" and "contradict" in result and "not contradict" not in result:
                token_idx, k = [x for x in enumerate(tokens) if "contradict" in x[-1]][0]
                correct += 1
            elif label == "entail" and "entail" in result and "not entail" not in result:
                token_idx, k = [x for x in enumerate(tokens) if "entail" in x[-1]][0]
                correct += 1
            elif label == "neutral" and ("irrelevant" in result or "does not entail" in result \
                    or "does not violate" in result or "does not contradict" in result):
                if "irrelevant" in result:
                    token_idx, k = [x for x in enumerate(tokens) if "irrelevant" in x[-1]][0]
                else:
                    token_idx, k = [x for x in enumerate(tokens) if "does" in x[-1]][0]
                correct += 1
            else:
                token_idx, k = [x for x in enumerate(tokens) if "contradict" in x[-1]\
                        or "entail" in x[-1] or "is" in x[-1]][0]
                wrong += 1
            result_prob = response["logprobs"][token_idx]
            k = [k for k, v in result_prob.items() if label_tok_map[label][0] in k]
            if len(k)==0:
                k = [k for k, v in result_prob.items() if label_tok_map[label][1] in k]
            k = k[0] if len(k)>0 else k
            try:
                right_ans_prob = math.exp(result_prob[k])
            except:
                k = " is"
                right_ans_prob = math.exp(-10)
                continue 
            wrong_ans_log, alt_k_found = -10, ""
            for alt_k, alt_v in result_prob.items():
                if alt_v>wrong_ans_log and k.lower() not in alt_k.lower() and alt_k.lower() not in k.lower():
                    if alt_k.rstrip().lstrip() not in ["<","entirely","nor","or","is","does","neither","is","show"] and \
                            not ("irrelevant" in k and "unrelated" in alt_k):
                        wrong_ans_log=alt_v
                        alt_k_found  = alt_k
            wrong_ans_prob = math.exp(wrong_ans_log)
            result_prob = right_ans_prob /(right_ans_prob+wrong_ans_prob)
            probs.extend([(1, result_prob),(0, 1-result_prob)]) 
            if label=="violate" or label=="entail":
                right_ans_prob = math.exp(max([v for k,v in response["logprobs"][token_idx].items() \
                            if "ent" in k.lower() or "cont" in k.lower()]))
                wrong_ans_logs = [v for k,v in response["logprobs"][token_idx].items() \
                            if "not" in k.lower() or "ir" in k.lower()]
                wrong_ans_logs.extend([v for k,v in response["logprobs"][token_idx+1].items() \
                            if "not" in k.lower()])
                wrong_ans_prob = math.exp(max(wrong_ans_logs)) if len(wrong_ans_logs)>0 else 0.0
            else:
                right_ans_logs = [v for k,v in response["logprobs"][token_idx].items() \
                            if "not" in k.lower() or "ir" in k.lower()]
                right_ans_logs.extend([v for k,v in response["logprobs"][token_idx+1].items() \
                            if "not" in k.lower()])
                right_ans_prob = math.exp(max(right_ans_logs)) if len(right_ans_logs)>0 else 0.0
                wrong_ans_logs = [v for k,v in response["logprobs"][token_idx].items() \
                            if "ent" in k.lower() or "cont" in k.lower()]
                wrong_ans_logs.extend([v for k,v in response["logprobs"][token_idx-1].items() \
                            if "ent" in k.lower() or "cont" in k.lower()])
                wrong_ans_prob = math.exp(max(wrong_ans_logs)) if len(wrong_ans_logs)>0 else 0.0
            result_prob = right_ans_prob /(right_ans_prob+wrong_ans_prob)
            two_cls_probs.extend([(1, result_prob),(0, 1-result_prob)])
            
    if "2" in results_fname:
        print("### Two-Class ###")
        print(c/(c+w))
        print(len(probs),roc_auc_score([x[0] for x in probs], [x[1] for x in probs]))
    else:
        print("### Three-Class ###")
        print(correct, wrong, correct/(correct+wrong))
        print(len(probs),roc_auc_score([x[0] for x in probs], [x[1] for x in probs]))

if __name__ == "__main__":
    #init_sample_n_annotate()

    results_fname = "data/eval_results/norm_grounding_speakerLevel_2.json"
    prompt(results_fname)
    eval(results_fname)

    results_fname = "data/eval_results/norm_grounding_speakerLevel_3.json"
    prompt(results_fname)
    eval(results_fname)
