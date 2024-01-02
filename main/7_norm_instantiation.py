"""
Speaker-Level Norm Instantiation Grounding Results
"""
from convokit import Corpus, download
import os, json, math, pandas as pd
import openai
from sklearn.metrics import roc_auc_score


openai.organization = os.getenv("OPENAI_ORG")
openai.api_key = os.getenv("OPENAI_API_KEY")

norms = ["One should always try to be polite and respectful to one's peers.",\
        "One should always try to be truthful and honest in one's reviews."]

corpus = Corpus(filename=download("conversations-gone-awry-corpus"))


with open("data/norm_discoveries/NormSage_wSchema.json","r") as f:
    GPT3norm = json.load(f)

def init_sample_n_annotate():
    pass
    #df.to_csv("../NormSage/data/human_assessment/GroundingAnnotationSpeakerLevel.csv")

def prompt():
    df = pd.read_csv("../NormSage/data/human_assessment/GroundingAnnotationSpeakerLevel.csv")
    responses = {}
    if os.path.exists("data/norm_grounding/speakerLevel_results.json"):
        with open("data/norm_grounding/speakerLevel_results.json", "r") as f:
            responses = json.load(f)
    for fname_idx, data in df.iterrows():
        fname = data["fname"].replace(".txt",".json")
        if fname not in responses:
            responses[fname] = {}
        with open("data/dialchunk/"+fname,"r") as f:
            dial = json.load(f)["txt"]
        for label_idx, label in zip([1,2,3],["violate","entail","neutral"]):
            if data["grounding"+str(label_idx)] != data["grounding"+str(label_idx)]:
                continue
            speaker_ln = data["grounding"+str(label_idx)].split("Norm")[0].split("(Dialogue:")[-1]
            speaker = speaker_ln.split(":")[0].lstrip()
            norm = data["grounding"+str(label_idx)].split("Norm: ")[-1].split(")")[0]
            prompt_txt = "Given the following dialogue:\n"
            prompt_txt += dial
            prompt_txt += "\n\nAnd given the social norm: "+norm+"\n\n"
            prompt_txt += "Explain whether what's spoken by "+str(speaker)+" in the conversation " + \
                          "<entails, is irrelevant with, or contradicts> the social norm and why."
            if True: #label not in responses[fname]:
                response = openai.Completion.create(
                  engine="text-davinci-002",
                  prompt = prompt_txt,
                  temperature=0,
                  logprobs=5,
                  max_tokens=80,
                  top_p=1
                )
                response = response["choices"][0]
                responses[fname][label] = {"norm":norm, "prompt_txt":prompt_txt, \
                        "grounding_result":response["text"].lstrip(), \
                        "tokens":response["logprobs"]["tokens"],"logprobs":response["logprobs"]["top_logprobs"]}
        with open("data/norm_grounding/speakerLevel_results_wProb.json", "w") as f:
            json.dump(responses, f)

def get_logprob_helper(logprob_dict, kw):
    if kw==" not":
        print(logprob_dict)
        return (max(list(logprob_dict[0][" not"].values()))+logprob_dict[1][" not"]+max(list(logprob_dict[2].values())))/3
    elif kw in logprob_dict:
        return logprob_dict[kw]
    elif kw+"s" in logprob_dict:
        return logprob_dict[kw+"s"]
    return None

def conf_score_helper(label, pred_label, tokens, logprobs):
    if " "+pred_label in tokens or " "+pred_label+"s" in tokens:
        token_idx, k = [x for x in enumerate(tokens) if (" "+pred_label in x[-1] or " "+pred_label+"s" in x[-1])][0]
        match_prob = logprobs[token_idx][k]
    else: #case of incorrect prediction
        print("AA",label, tokens)
        for prob in logprobs:
            print(prob)
        token_idx, k = [x for x in enumerate(tokens) if (" entails" in x[-1] or \
                " entail" in x[-1] or " contradicts" in x[-1] or " contradict" in x[-1])][0]
        if tokens[token_idx-1]==" not":
            match_prob = logprobs[token_idx-2][" "+pred_label+"s"]
        elif " "+pred_label in logprobs[token_idx]:
            match_prob = logprobs[token_idx][" "+pred_label]
        elif " "+pred_label+"s" in logprobs[token_idx]:
            match_prob = logprobs[token_idx][" "+pred_label+"s"]
        else:
            match_prob = -999
    alt_ans_prob = None
    if pred_label=="entail":
        alt_ans_prob = get_logprob_helper(logprobs[token_idx]," contradict")
    elif pred_label=="contradict":
        alt_ans_prob = get_logprob_helper(logprobs[token_idx]," entail")
    elif pred_label=="irrelevant" or pred_label=="not":
        alt_ans_prob = get_logprob_helper(logprobs[token_idx-1]," entail")
        alt_ans_prob = get_logprob_helper(logprobs[token_idx-1]," contradict")
    if alt_ans_prob is None:
        if " not" in tokens and pred_label!="entail" and pred_label!="contradict":
            token_idx, k = [x for x in enumerate(tokens) if " not" in x[-1]][0]
            alt_ans_prob = get_logprob_helper(logprobs[token_idx-1:token_idx+2]," not")
        else:
            alt_ans_prob = -999
    match_prob, alt_ans_prob = math.exp(match_prob), math.exp(alt_ans_prob)
    return match_prob/(match_prob+alt_ans_prob)

def eval():
    with open("data/norm_grounding/speakerLevel_results_wProb.json", "r") as f:
        NormSage = json.load(f)
    correct, wrong = 0, 0
    auc, auc_entail, auc_violate = [], [], []
    for fname, v in NormSage.items():
        for label_idx, label in zip([1,2,3],["violate","entail","neutral"]):
            if label not in NormSage[fname]:
                continue
            result = NormSage[fname][label]["grounding_result"]
            tokens = NormSage[fname][label]["tokens"]
            response = NormSage[fname][label]
            if label == "violate" and "contradict" in result and "not contradict" not in result:
                result_prob = conf_score_helper(label, "contradict", tokens, response["logprobs"])
                auc_violate.append((1, result_prob))
                correct += 1
            elif label == "entail" and "entail" in result and "not entail" not in result:
                result_prob = conf_score_helper(label, "entail", tokens, response["logprobs"])
                auc_entail.append((1, result_prob))
                correct += 1
            elif label == "neutral" and ("irrelevant" in result or "does not" in result):
                if " irrelevant" in tokens[:tokens.index(" not")]:
                    result_prob = conf_score_helper(label, "irrelevant", tokens, response["logprobs"])
                else:
                    result_prob = conf_score_helper(label, "not", tokens, response["logprobs"])
                auc_entail.append((0, 1-result_prob))
                auc_violate.append((0, 1-result_prob))
                correct += 1
            else:
                wrong += 1
                probe_label = "contradict" if label == "violate" else "irrelevant" if label=="neutral" else label
                result_prob = conf_score_helper(label, probe_label, tokens, response["logprobs"])
                if label == "entail":
                    auc_entail.append((1, result_prob))
                elif label == "violate":
                    auc_violate.append((1, result_prob))
                elif label == "neutral":
                    auc_entail.append((0, 1-result_prob))
                    auc_violate.append((0, 1-result_prob))
    print(correct, wrong)
    # Compute AUC scores to evaluate the confidence thresholds
    print(auc_entail)
    auc = list(set(auc_entail + auc_violate))
    auc = roc_auc_score([x[0] for x in auc], [x[1] for x in auc])
    auc_entail = roc_auc_score([x[0] for x in auc_entail], [x[1] for x in auc_entail])
    print(auc_violate)
    auc_violate = roc_auc_score([x[0] for x in auc_violate], [x[1] for x in auc_violate])
    print(auc, auc_entail, auc_violate)

#init_sample_n_annotate()
#prompt()
eval()

