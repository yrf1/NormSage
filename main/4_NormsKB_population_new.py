"""
NormsKB schema:
normsID, foundations, primary topic, secondary topic, culture, norm description, 
entail example, provenance, speaker, explanation
contradict example, provenance, speaker, explanation
"""
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from datetime import datetime
import os, json, torch, pickle
import openai
import copy


device = "cuda" if torch.cuda.is_available() else "cpu"

openai.organization = os.getenv("OPENAI_ORG")
openai.api_key = os.getenv("OPENAI_API_KEY")

def save_file(data, fname, fmt="pkl"):
    if fmt=="json":
        with open(fname, "w") as f:
            json.dump(data, f)
    else: 
        with open(fname, "wb") as f:
            pickle.dump(data, f)

def cache_embed(source_list, norm_list, save_fname):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").eval().to(device)
    norm_idx2embed_cache = {}
    if os.path.exists(save_fname):
         with open(save_fname, "rb") as f:
            norm_idx2embed_cache = pickle.load(f)
    for i, norm in enumerate(norm_list):
        if norm in norm_idx2embed_cache:
            continue
        tok = tokenizer(norm.replace("It's","It is"),return_tensors="pt")
        tok = {k: v.to(device=device) for k, v in tok.items()}
        embed = model(**tok).pooler_output
        norm_idx2embed_cache[norm] = embed.detach().cpu()
        if i % 100 == 0:
            save_file(norm_idx2embed_cache, save_fname)
    save_file(norm_idx2embed_cache, save_fname)
    return norm_idx2embed_cache

def deduplicate(source_list, norm_list, norm_idx2embed_cache, save_fname):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    normsKB, skip_idx = {}, []
    if os.path.exists(save_fname):
        with open(save_fname, "rb") as f:
            normsKB = pickle.load(f)
        normsKB_norm_lst = [v["norm"] for (k,v) in normsKB.items()]
        skip_idx = [idx for idx, norm in enumerate(norm_list) if norm in normsKB_norm_lst]
    for i1, norm1 in enumerate(norm_list):
        if i1 not in skip_idx:# and not any(s in norm1 for s in speakers):
            normsKB[len(normsKB)] = {"source": source_list[i1], "norm":norm1} 
            skip_idx.append(i1)
        for i2, norm2 in enumerate(norm_list):
            if i2 > i1 and i2 not in skip_idx:
                embed1, embed2 = norm_idx2embed_cache[norm1], norm_idx2embed_cache[norm2]
                cos_score = cos(embed1, embed2).item()
                if cos_score <= 0.998:# and not any(s in norm2 for s in speakers):
                        normsKB[len(normsKB)] = {"source": source_list[i2], "norm":norm2} 
                skip_idx.append(i2)
        if i1 % 100 == 0:
            save_file(normsKB, save_fname)
    save_file(normsKB, save_fname)
    return normsKB

def check_correctness(normsKB_old, save_fname="", fname_list=None):
    normsKB = {}
    if os.path.exists(save_fname):
        with open(save_fname, "r") as f:
            normsKB = json.load(f)
    for idx, (k, v) in enumerate(normsKB_old.items()):
        if fname_list!=None and k not in fname_list:
            continue
        print(len(normsKB))
        if k not in normsKB:
            normsKB[k] = v
            normsKB[k]["correctness"] = []
            normsKB[k]["correctness_tokens_n_logprobs"] = []
        if len(normsKB[k]["correctness"]) == len(normsKB[k]["norms"]):
            continue
        for norm_idx, norm in enumerate(v["norms"]):
            if norm_idx < len(normsKB[k]["correctness"]):
                continue
            norm = norm[-1] if len(norm)==2 else norm
            prompt_Q = "Consider this sentence:\n"+norm
            prompt_Q += '\n\nIs this a correct/acceptable social norm? Answer "yes" or "no", and then explain why.'
            response = openai.Completion.create(
              engine="text-davinci-003",
              prompt = prompt_Q,
              temperature=0.0,max_tokens=256,logprobs=5)
            normsKB[k]["correctness"].append(response["choices"][0]["text"].lstrip())
            normsKB[k]["correctness_tokens_n_logprobs"].append((response["choices"][0]["logprobs"]["tokens"], \
                                                response["choices"][0]["logprobs"]["top_logprobs"]))
        if idx % 2 == 0:
            save_file(normsKB, save_fname, fmt="json")
    save_file(normsKB, save_fname, fmt="json")
    return normsKB

def check_relevance(norms, save_fname="", fname_list=None):
    norms_wSelfGround = {}
    if os.path.exists(save_fname):
        with open(save_fname, "r") as f:
            norms_wSelfGround = json.load(f)
    for idx, (k, v) in enumerate(norms.items()):
        if k not in fname_list:
            continue
        if k not in norms_wSelfGround:
            norms_wSelfGround[k] = v
            norms_wSelfGround[k]["grounding_result"] = []
            norms_wSelfGround[k]["grounding_tokens_n_logprobs"] = []
        if len(norms_wSelfGround[k]["grounding_result"]) == len(norms_wSelfGround[k]["norms"]):
            continue
        print(save_fname, len(norms_wSelfGround))
        if os.path.exists("data/dialchunk/"+k):
            with open("data/dialchunk/"+k, "r") as f:
                dial = json.load(f)["txt"]
            for norm_idx, norm in enumerate(v["norms"]):
                if norm_idx < len(norms_wSelfGround[k]["grounding_result"]) or (len(norm)==2 and len(norm[-1])<4):
                    continue
                if "yes" not in v["correctness"][norm_idx].lower() and "no" in v["correctness"][norm_idx].lower():
                    norms_wSelfGround[k]["grounding_result"].append(None)
                    norms_wSelfGround[k]["grounding_tokens_n_logprobs"].append(None)
                    continue
                norm = norm[-1] if len(norm)==2 else norm
                prompt_txt = "Given the following dialogue:\n"
                prompt_txt += dial
                prompt_txt += "\n\nAnd given the social norm: "+norm+"\n\n"
                prompt_txt += "Explain whether what's spoken in the conversation " + \
                          "<entails, is irrelevant with, or contradicts> the social norm and why."
                response = openai.Completion.create(
                  engine="text-davinci-003",
                  prompt = prompt_txt,logprobs=5,
                  temperature=0.0, max_tokens=256, top_p=1)
                response = response["choices"][0]
                norms_wSelfGround[k]["grounding_result"].append(response["text"].lstrip())
                norms_wSelfGround[k]["grounding_tokens_n_logprobs"].append(\
                        (response["logprobs"]["tokens"], response["logprobs"]["top_logprobs"]))
        save_file(norms_wSelfGround, save_fname, fmt="json")

def norm_dvr_update(norms_wSchema, init_norm_dvr_fname):
    with open(init_norm_dvr_fname.replace(".json","_cor_rlv.json"), "r") as f:
        norms_discovery_data = json.load(f)
    for k, v in copy.deepcopy(norms_discovery_data).items():
        if "correctness" not in v:
            pass
        if len(v["correctness"]) != len(v["norms"]):
            continue
        norms_discovery_data[k]["norms_final"] = []
        for norm_idx, norm in enumerate(v["norms"]):
            if v["correctness"][norm_idx].split()[0] == "Yes," and \
                    (("entails" in v["grounding_result"][norm_idx] and \
                    "does not entail" not in v["grounding_result"][norm_idx]) or \
                    ("contradicts" in v["grounding_result"][norm_idx] and \
                    "does not contradict" not in v["grounding_result"][norm_idx])):
                norms_discovery_data[k]["norms_final"].append(norm)
    with open(init_norm_dvr_fname.replace(".json","_filt_final.json"), "w") as f:
        json.dump(norms_discovery_data, f)


def categorize(normsKB):
    with open("data/tmpl/norm_categories.txt", "r") as f:
        tmpl = f.read()
    foundations, topics = tmpl.split("----")
    foundations, topics = foundations.split("\n"), topics.split("\n")
    topics = [x for x in topics if len(x)>0]
    tokenizer = AutoTokenizer.from_pretrained("bigscience/T0pp")
    model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp")
    for k, v in normsKB.items():
        norm = v["norm"]
        prompt_Q = "Given the norm:\n"+norm
        prompt_Q += '\n\nIs this about "(i) social communication", "(ii) human behavior", \
                    "(iii) concepts and beliefs", or "(iv) something else"?'
        x = tokenizer.encode(prompt_Q,return_tensors="pt")
        y = model.generate(x, num_beams=3, top_p=0.92)
        prompt_A = tokenizer.decode(y[0],skip_special_tokens=True)
        normsKB[k]["category"] = prompt_A
        topic_and_score = []
        for topic in topics:
            prompt_Q = "Given the norm:\n"+norm+'\n\nIs this norm related to the topic of "'+topic+'"'
            x = tokenizer.encode(prompt_Q,return_tensors="pt")
            y = model.generate(x, num_beams=3, top_p=0.92,output_scores=True,return_dict_in_generate=True)
            tok, prob = y.sequences, y.sequences_scores
            prompt_A = tokenizer.decode(tok[0],skip_special_tokens=True)
            topic_and_score.append((topic, prompt_A, prob))
        normsKB[k]["primary_topic"] = [topic for (topic, prompt_A, prob) in topic_and_score if "no" not in prompt_A.lower()]
        normsKB[k]["primary_topic"] = normsKB[k]["primary_topic"][0] if len(normsKB[k]["primary_topic"])>0 else ""
        topic_and_score = [x for x in topic_and_score if x[0] != normsKB[k]["primary_topic"]]
        normsKB[k]["secondary_topic"] = [topic for (topic, prompt_A, prob) in topic_and_score if "no" not in prompt_A.lower()]
        normsKB[k]["secondary_topic"] =  normsKB[k]["secondary_topic"][0] if len(normsKB[k]["secondary_topic"])>0 else ""
    return normsKB

def main(init_norm_dvr_fname, fname_list=None):
    # Self-Verification
    with open(init_norm_dvr_fname, "r") as f:
        norms_wSchema = json.load(f)
    check_correctness(norms_wSchema, init_norm_dvr_fname.replace(".json","_cor.json"), fname_list)
    with open(init_norm_dvr_fname.replace(".json","_cor.json"), "r") as f:
        norms_wSchema = json.load(f)
    check_relevance(norms_wSchema, init_norm_dvr_fname.replace(".json","_cor_rlv.json"), fname_list) 
    norm_dvr_update(norms_wSchema, init_norm_dvr_fname)
    """"data/norm_discoveries/NormsKB_intmd_rlv_check.json")
    # Self-Verification
    intmd_fname = "data/norm_discoveries/NormsKB_intmd_corr_check.json"
    norms_wSchema_corr = check_correctness(norms_wSchema, intmd_fname)
    intmd_fname = "data/norm_discoveries/NormsKB_intmd_rlv_check.json"
    norms_wSchema_rlv = check_relevance(norms_wSchema, intmd_fname)"""
    # De-Duplication
    source_list, norm_list = [], []
    for k, v in norms_wSchema.items():
        for norm in v["norms"]:
            norm = norm[-1]
            if norm not in norm_list and norm.rstrip().split()[-1]!="include:" \
                    and norm.rstrip().split()[-1]!="be:" and norm.rstrip().split()[-1]!="follows:":
                source_list.append(k) 
                norm_list.append(norm[3:] if norm[1:3]==". " else norm)
    intmd_fname = "data/norm_discoveries/NormsKB_intmd_embed.pkl"
    norm_idx2embed_cache = cache_embed(source_list, norm_list, intmd_fname)    
    intmd_fname = "data/norm_discoveries/NormsKB_intmd_dedup.pkl"
    normsKB = deduplicate(source_list, norm_list, norm_idx2embed_cache, intmd_fname)
    with open(intmd_fname, "rb") as f:
        normsKB = pickle.load(f)
    #print(len(normsKB))
    #intmd_fname = "data/norm_discoveries/NormsKB_intmd_corr_check.json"
    #normsKB = check_correctness(normsKB, intmd_fname)
    #print(len(normsKB))
    #normsKB = {k:v for k,v in normsKB.items() if "no" not in v["correctness"].split()[0]}
    #print(len(normsKB))
    normsKB = categorize(normsKB)
    with open("data/norm_discoveries/NormsKB.json", "w") as f:
        json.dump(normsKB, f)
    return

fname_list = None
with open("data/human_assessment/fID_list_100.txt", "r") as f:
    fname_list = f.read().split("\n")
    fname_list = [x+".json" for x in fname_list if len(x)>0]
main("data/norm_discoveries/NormSage_base.json", fname_list)
main("data/norm_discoveries/NormSage_wSchema_new.json", fname_list)
main("data/norm_discoveries/NormSage_wIndc.json", fname_list)

