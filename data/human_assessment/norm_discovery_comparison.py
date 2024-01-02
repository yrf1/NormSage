"""Current MTurk UI Project Title: 'NormComparisonQual'"""
import os, csv, json
import numpy as np, pandas as pd
import collections, random
import boto3, openai


QUESTIONS_v1 = ["relevance","relatableness","well-formedness","correctness","insightfulness","diversity"]
QUESTIONS_v2 = ["is the norm inspired from the situation, and how well does the norm apply to the situation",\
                "does the norm balance vagueness against specificity, so that it can generalize across multiple situations "+\
                "without being too specific (e.g., 'it is rude not to share your mac and chesse with your younger brother')",\
                "is the norm self-contained (description is clear), and is the norm well-structured (contains both a judgement of acceptability and assessed beheavior)",\
                "do you agree that the described norm holds to the best of your knowledge",\
                "does the norm convey something interesting and helpful for situations similar to the dialogue scenario you read at the top of the page",\
                "do the norms cover breadth for different aspects that could be applicable in the scenario"]

def read_source_data():
    with open("data/norm_discoveries/baseline_SocialChem_rtrv_1.json","r") as f:
        ROT_rtv_all = json.loads(f.read())
    with open("data/norm_discoveries/baseline_SocialChem_gen.json","r") as f:
        ROT_gen = json.loads(f.read())
    with open("data/norm_discoveries/baseline_MIC_rtrv.json", "r") as f:
        MIC_rtv = json.load(f)
    with open("data/norm_discoveries/baseline_T0_norm_discovery_new.json", "r") as f:
        T0_gen = json.load(f,object_pairs_hook=collections.OrderedDict)
    with open("data/norm_discoveries/baseline_MIC_gen_all_proc.json", "r") as f:
        MIC_gen = json.load(f)
    with open("data/norm_discoveries/NormSage_base.json","r") as f:
        GPT3 = json.load(f)
    with open("data/norm_discoveries/NormSage_base_filt_final.json","r") as f:
        NormSage = json.load(f)
    with open("data/norm_discoveries/NormSage_wSchema_new_filt_final.json","r") as f:
        NormSage_plus_wSchema = json.load(f)
    with open("data/norm_discoveries/NormSage_wIndc.json","r") as f:
        NormSage_plus_wIndc = json.load(f)
    with open("data/norm_discoveries/NormSage_mini.json","r") as f:
        NormSage_mini = json.load(f)
    keys_union = list(set(list(ROT_rtv_all.keys())).intersection(list(MIC_rtv.keys())).intersection(list(MIC_gen.keys())))
    keys_union = list(set(keys_union).intersection(list(ROT_gen.keys())).intersection(list(GPT3.keys())).intersection(list(T0_gen.keys())))
    keys_union = list(set(keys_union).intersection(list(NormSage.keys()))) #.intersection([x+".json" for x in list(GPT3norm2.keys())]))
    return ROT_rtv_all, ROT_gen, MIC_rtv, MIC_gen, T0_gen, GPT3, NormSage, \
            NormSage_mini, NormSage_plus_wSchema, NormSage_plus_wIndc, keys_union

def submitBatch(mode="API"):
    ROT_rtv_all, ROT_gen, MIC_rtv, MIC_gen, T0_gen, GPT3, \
            NormSage, NormSage_mini, NormSage_plus_wFrame, NormSage_wIndc, keys_union = read_source_data()
    MTurkBatchInput_header = [("input","norm_metadata","norm_list","norms_rate_1","norms_rate_2","norms_rate_3","norms_rate_4","norms_rate_5")]
    with open("data/human_assessment/fID_list_100.txt", "r") as f:
        keys_union = [x for x in f.read().split("\n") if len(x)>0]
    data = ROT_rtv_all, ROT_gen, MIC_rtv, MIC_gen, T0_gen, GPT3, NormSage, NormSage_mini, NormSage_plus_wFrame
    #autoAnalyzeResults(keys_union, data) 
    print(len(keys_union))
    count = 0
    for fname_idx, fname in enumerate(keys_union):
        fname = fname#+".json"
        if fname in NormSage_wIndc or fname+".json" in NormSage_wIndc: #T0_gen: #GPT3norm0: #fname in MIC_rtv \
                #and fname in ROT_gen and fname in MIC_gen \
                #and fname in GPT3norm0 and fname in GPT3norm \
                #: #and fname in GPT3norm2:
            count += 1
    for fname_idx, fname in enumerate(keys_union):
        fname = fname+".json"
        MTurkBatchInput = []
        with open("data/dialchunk/"+fname,"r") as f:
            dial = f.read() 
            dial = dial if '"txt":' not in dial else json.loads(dial)["txt"]
        if fname in ROT_rtv_all:
            ROTrtv = ROT_rtv_all[fname]["norm"]
            for norm in ROTrtv[:3]:
                MTurkBatchInput.append((fname, "SocialChem_rtv", norm, "", "", "", "", ""))
        for norm in ROT_gen[fname][:3]:
            norm = norm.replace("[rot]","")
            MTurkBatchInput.append((fname, "NMT_gen", norm, "", "", "", "", ""))
        for norm in MIC_rtv[fname]["norm"][:3]:
            MTurkBatchInput.append((fname, "MIC_rtrv", norm, "", "", "", "", ""))
        for norm in MIC_gen[fname]["norms"]:
            MTurkBatchInput.append((fname, "MIC_gen", norm, "", "", "", "", ""))
        for norm in T0_gen[fname]["norm"][:3]:
            MTurkBatchInput.append((fname, "T0", norm, "", "", "", "", ""))
        for norm in NormSage_mini[fname][:3]:
            norm = norm[3:] if norm[1:2]==". " else norm
            MTurkBatchInput.append((fname, "NormSage_mini_local", norm, "", "", "", "", ""))
        for norm in ROTrtv[:3]:
            MTurkBatchInput.append((fname, "ROTrtv", norm, "", "", "", "", ""))
        if fname in GPT3:
            for norm in GPT3[fname]["norms"][:3]:
                norm = norm[3:] if norm[1:3]==". " else norm
                MTurkBatchInput.append((fname, "GPT3", norm, "", "", "", "", ""))
        if fname in NormSage:
            for norm in NormSage[fname]["norms_final"][:3]:
                MTurkBatchInput.append((fname, "NormSage", norm, "", "", "", "", ""))
        if fname in NormSage_plus_wFrame:
            GPT3gen = NormSage_plus_wFrame[fname]["norms_final"]
            for norm in GPT3gen[:3]:
                if type(norm) is list and len(norm)==2:
                    norm = norm[-1]
                MTurkBatchInput.append((fname, "NormSage_plus_wSchema", norm, "", "", "", "", ""))
        if fname.split(".json")[0] in NormSage_wIndc:
            GPT3gen2 = NormSage_wIndc[fname.split(".json")[0]]["norms"]
            GPT3gen2 = [(x if x[1:3]!=". " else x[3:]) for x in GPT3gen2]
            GPT3gen2_old = GPT3gen2
            GPT3gen2 = []
            saved, saved_tmpl="", ""
            for x_idx, x in enumerate(GPT3gen2_old):
                if x.rstrip()[-9:]==" culture:":
                    saved, saved_tmpl = x, NormSage_wIndc[fname.split(".json")[0]]["Qs"][x_idx]
                else:
                    if NormSage_wIndc[fname.split(".json")[0]]["Qs"][x_idx]==saved_tmpl:
                        if saved.split()[0] not in x:
                            x = saved+" "+x
                    else:
                        saved = ""
                    GPT3gen2.append(x)
            for norm in GPT3gen2[:3]:
                MTurkBatchInput.append((fname, "NormSage_wIndc", norm, "", "", "", "", ""))
    #MTurkBatchInput_header, MTurkBatchInput_data = MTurkBatchInput[0], MTurkBatchInput[1:]
        random.shuffle(MTurkBatchInput)
        MTurkBatchInput = MTurkBatchInput_header + MTurkBatchInput
        MTurkBatchInput[1] = (fname,)+MTurkBatchInput[1][1:]
        MTurkBatchInput[3] = (dial,)+MTurkBatchInput[3][1:]
        print("~~~",len(MTurkBatchInput))
        
        for q_idx in range(5):
            q_cat = QUESTIONS_v1[q_idx]
            MTurkBatchInput[2] = (QUESTIONS_v2[q_idx],)+MTurkBatchInput[2][1:]
            with open("MTurk/neww/"+q_cat+"/norm_comparison_worksheet"+"_dial_idx_"+str(fname_idx)+".csv", "w", encoding="utf-8") as f:
                print(len(MTurkBatchInput))
                writer = csv.writer(f)
                writer.writerows(MTurkBatchInput)
            with open("MTurk/neww/"+q_cat+"/norm_comparison_masked"+"_dial_idx_"+str(fname_idx)+".csv", "w", encoding="utf-8") as f:
                masked_MTurkBatchInput = [(x[0],"")+x[2:] for x in MTurkBatchInput]
                writer = csv.writer(f)
                writer.writerows(masked_MTurkBatchInput)
        
def evalBatchJan31():
    for q_idx in range(6):
        if q_idx<5:
            q_cat = QUESTIONS_v1[q_idx]
        else:
            q_cat = "overall"
        print(q_cat, "~~~~~~~~~~~~~")
        tracker = {}
        for fname in os.listdir("MTurk/new/"+q_cat):
            if "mask" in fname:
                fname_unmasked = fname.replace("_masked_dial","_worksheet_dial")
                df = pd.read_csv("MTurk/new/"+q_cat+"/"+fname)
                if df[df["norms_rate_5"].notnull()].shape[0]==0:
                    continue
                df_unmasked = pd.read_csv("MTurk/new/"+q_cat+"/"+fname_unmasked)
                for idx, data in df.iterrows():
                    method = df_unmasked.iloc[idx]["norm_metadata"]
                    if method not in tracker:
                        tracker[method] = []
                    score = 5 if type(data["norms_rate_5"]) is str else \
                            4 if type(data["norms_rate_4"]) is str  else \
                            3 if type(data["norms_rate_3"]) is str  else \
                            2 if type(data["norms_rate_2"]) is str else \
                            1 #if data["norms_rate_1"]!= data["norms_rate_1"]
                    #print(idx, method, score, type(data["norms_rate_5"]) is str)
                    tracker[method].append(score)
        if len(tracker)>1:
            for k, v in tracker.items():
                if q_cat!="overall":
                    print(k, sum(v)/len(v), len(v))
                else:
                    print(k, v.count(5)/len(v), len(v))

def prompt_qual_rating(dial, norms):
    rating, avg_rating, expl = [[],[],[],[],[]], [], [[],[],[],[],[]]
    context_header = ""
    for norm in norms:
        conv_header = "Given the following conversation:\n\n"
        norm_header = "Consider this social/moral/socio-cultural norm:\n"
        # Relevance
        q_type = "Relevance"
        q_txt = "Rate on a scale of 1 to 5 how relevant this social norm is to the conversation:"
        if len(norm)==2:
            norm = norm[-1]
        prompt_txt = conv_header+dial+"\n\n"+norm_header+norm+"\n\n"+q_txt
        response = openai.Completion.create(engine="text-davinci-003",
          prompt = prompt_txt, temperature=0.0, max_tokens=256)
        response_txt = response["choices"][0]["text"].lstrip()
        this_rating = 5 if "5" in response_txt else 4 if "4" in response_txt \
                else 3 if "3" in response_txt else 2 if "2" in response_txt else 1
        rating[0].append(this_rating)
        expl[0].append(response_txt)
        # Well-Formedness
        q_type = "Well-Formedness"
        q_txt = "Rate on a scale of 1 to 5 whether the norm is well-formed in sentence structure. 5 means it's a full sentence with semantic framing that explicitly judge the content. 1 means it's some simple noun or verb clause, without characterization."
        prompt_txt = norm_header+norm+"\n\n"+q_txt
        response = openai.Completion.create(engine="text-davinci-003",
          prompt = prompt_txt, temperature=0.0, max_tokens=256)
        response_txt = response["choices"][0]["text"].lstrip()
        this_rating = 5 if "5" in response_txt else 4 if "4" in response_txt \
                else 3 if "3" in response_txt else 2 if "2" in response_txt else 1
        rating[1].append(this_rating)
        expl[1].append(response_txt)
        # Correctness
        q_type = "Correctness"
        q_txt = "Rate on a scale of 1 to 5 how correct this social norm is:"
        prompt_txt = norm_header+norm+"\n\n"+q_txt
        response = openai.Completion.create(engine="text-davinci-003",
          prompt = prompt_txt, temperature=0.0, max_tokens=256)
        response_txt = response["choices"][0]["text"].lstrip()
        this_rating = 5 if "5" in response_txt else 4 if "4" in response_txt \
                else 3 if "3" in response_txt else 2 if "2" in response_txt else 1
        rating[2].append(this_rating)
        expl[2].append(response_txt)
        # Insightfulness
        q_type = "Insightfulness"
        q_txt = "Rating on a scale of 1 to 5 how insightful this social norm is to the conversation?"
        prompt_txt = conv_header+dial+"\n\n"+norm_header+norm+"\n\n"+q_txt
        response = openai.Completion.create(engine="text-davinci-003",
          prompt = prompt_txt, temperature=0.0, max_tokens=256)
        response_txt = response["choices"][0]["text"].lstrip()
        this_rating = 5 if "5" in response_txt else 4 if "4" in response_txt \
                else 3 if "3" in response_txt else 2 if "2" in response_txt else 1
        rating[3].append(this_rating)
        expl[3].append(response_txt)
        # Relatableness
        q_type = "Relatableness"
        q_txt = "Rate on a scale of 1-5 how relatable this social norm is, and explain why. If it's too specific about a particular person or event, or too general, rate something like a 1. If it's a good reference for guiding human behavior, rate something like a 5."
        prompt_txt = norm_header+norm+"\n\n"+q_txt
        response = openai.Completion.create(engine="text-davinci-003",
          prompt = prompt_txt, temperature=0.0, max_tokens=256)
        response_txt = response["choices"][0]["text"].lstrip()
        this_rating = 5 if "5" in response_txt else 4 if "4" in response_txt \
                else 3 if "3" in response_txt else 2 if "2" in response_txt else 1
        rating[4].append(this_rating)
        expl[4].append(response_txt)
    avg_rating = [sum(x)/len(x) for x in rating]
    return rating, avg_rating, expl

def autoAnalyzeResults(fname_list, data):
    ROT_rtv_all, ROT_gen, MIC_rtv, MIC_gen, T0_gen, GPT3norm0, GPT3norm_loc, GPT3norm, GPT3norm2 = data
    auto_qual_rating, result_fname = {}, "data/human_assessment/NormComparisons/auto_results.json"
    if os.path.exists(result_fname):
        with open(result_fname, "r") as f:
            auto_qual_rating = json.load(f)
    """for fname in fname_list:
        print("# proc files: ", len(auto_qual_rating))
        fname = fname+".json"
        with open("data/dialchunk/"+fname,"r") as f:
            dial = f.read()
            dial = dial if '"txt":' not in dial else json.loads(dial)["txt"]
        if fname not in auto_qual_rating:
            auto_qual_rating[fname] = {}
        if "ROT_rtv" not in auto_qual_rating[fname]:
            if fname in ROT_rtv_all:
                norms = ROT_rtv_all[fname]["norm"][:3]
                rating, avg_rating, expl = prompt_qual_rating(dial, norms)
                auto_qual_rating[fname]["ROT_rtv"] = (norms, rating, avg_rating, expl)
        if "ROT_gen" not in auto_qual_rating[fname]:
            if fname in ROT_gen:
                norms = [x.replace("[rot]","") for x in ROT_gen[fname][:3]]
                rating, avg_rating, expl = prompt_qual_rating(dial, norms)
                auto_qual_rating[fname]["ROT_gen"] = (norms, rating, avg_rating, expl)
        if "MIC_rtv" not in auto_qual_rating[fname]:
            if fname in MIC_rtv:
                norms = MIC_rtv[fname]["norm"][:3]
                rating, avg_rating, expl = prompt_qual_rating(dial, norms)
                auto_qual_rating[fname]["MIC_rtv"] = (norms, rating, avg_rating, expl)
        if "MIC_gen" not in auto_qual_rating[fname]:
            if fname in MIC_gen:
                norms = MIC_gen[fname]["norms"][:3]
                rating, avg_rating, expl = prompt_qual_rating(dial, norms)
                auto_qual_rating[fname]["MIC_gen"] = (norms, rating, avg_rating, expl)
        if "T0_gen" not in auto_qual_rating[fname]:
            if fname in T0_gen:
                norms = T0_gen[fname]["norm"][:3]
                rating, avg_rating, expl = prompt_qual_rating(dial, norms)
                auto_qual_rating[fname]["T0_gen"] = (norms, rating, avg_rating, expl)
        if "NormSage_mini" not in auto_qual_rating[fname]:
            if fname in GPT3norm_loc:
                print('eh')
                norms = [GPT3norm_loc[fname][-1]] #["norms"][:3]
                rating, avg_rating, expl = prompt_qual_rating(dial, norms)
                auto_qual_rating[fname]["NormSage_mini"] = (norms, rating, avg_rating, expl)
        if "NormSage_base" not in auto_qual_rating[fname]:
            if fname in GPT3norm0:
                norms = GPT3norm0[fname]["norms"][:3]
                rating, avg_rating, expl = prompt_qual_rating(dial, norms)
                auto_qual_rating[fname]["NormSage_base"] = (norms, rating, avg_rating, expl)
        if "NormSage_frame" not in auto_qual_rating[fname]:
            if fname in GPT3norm:
                norms = GPT3norm[fname]["norms"][:3]
                rating, avg_rating, expl = prompt_qual_rating(dial, norms)
                auto_qual_rating[fname]["NormSage_frame"] = (norms, rating, avg_rating, expl)
        if "NormSage_wIndc" not in auto_qual_rating[fname]:
            if fname in GPT3norm2:
                norms = GPT3norm2[fname]["norms"][:3]
                rating, avg_rating, expl = prompt_qual_rating(dial, norms)
                auto_qual_rating[fname]["MIC_gen"] = (norms, rating, avg_rating, expl)
        with open(result_fname, "w") as f:
            json.dump(auto_qual_rating, f)"""
    result_tracker = {}
    for k, v in auto_qual_rating.items():
        for k2, v2 in v.items():
            if k2 not in result_tracker:
                result_tracker[k2] = []
            result_tracker[k2].append(v2[2])
    print("relevance", "well-formed", "correct","insightful","relatable")
    for k, v in result_tracker.items():
        print(k, np.mean(np.array(v), axis=0))
    return

def analyzeResults(data_dir):
    for question_keyword in QUESTIONS_v1:
        print(question_keyword)
        final_a, final_b, final_c = [], [], []
        results = {}
        for fname in os.listdir(data_dir):
            df = pd.read_csv(data_dir+"/"+fname)
            if "batch" not in fname: 
                continue
            #print(fname, df.columns)
            df = df[df["AssignmentStatus"]!="Rejected"]
            for idx, data in df.iterrows():
                if question_keyword not in data['Input.question_v1']:
                    continue
                for norm_idx, normID in enumerate((data['Input.norm_order'] \
                        if "--" not in data['Input.norm_order'] \
                        else data['Input.norm_order'].split("--"))):
                    if normID not in results:
                        results[normID] = []
                    answers = [x for x in list(data.keys()) if "Answer.howMuch" in x and str(norm_idx+1) in x][0]
                    results[normID].append(data[answers])
        for k, v in results.items():
            print(k, sum(v)/len(v), "# participants: ",len(v))
        print()
    return

submitBatch("API")
#evalBatchJan31()
#submitBatch("Traditional")
#analyzeResults("data/human_assessment/NormComparisons")
