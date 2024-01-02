"""
What are some cultural norm violations in discussions about books
"""

import os
import json
import openai


openai.organization = os.getenv("OPENAI_ORG") #"org-fGi4Hgnq7sCektnzHAWEHeSi"
openai.api_key = os.getenv("OPENAI_API_KEY")

results = {}

data_dir = "data/dialchunk" #prompt/indicators"
for fname in [x for x in os.listdir(data_dir) if ".json" in x]:
    if "FreshOffTheBoat_S1_EP1" not in fname and "Blackish_S1_EP1" not in fname \
            and "Never-Have-I-Ever-S01E01" not in fname and "Citizen_Khan_S1_EP1" not in fname \
            and "AmericanFactory_" not in fname and \
            "Outsourced_S1_EP1_" not in fname:
        continue
    with open("data/dialchunk/"+fname, "r") as f:
        dial = json.load(f)["txt"]
    if not (os.path.exists("data/prompt/indicators/"+fname)): # or \
        continue
    with open("data/prompt/indicators/"+fname, "r") as f:
        indcs = f.read().split("talking about ")[-1] 
    with open("data/prompt/indictors_supplementary/results.json", "r") as f:
        uh = json.load(f)
        indcs_more_fname = fname.split("_")[0]+"_Khan.txt" if "Citizen" in fname.split("_")[0] \
                else fname.split("_")[0]+".txt"
        if indcs_more_fname not in uh:
            continue
        indcs_more = dict(uh[indcs_more_fname])
    loc = indcs_more["What location is this situation based in?"]
    culture = indcs_more["What are the cultures or ethnicities involved here?"]
    culture1, culture2 = culture.split(" and ")
    prompt = "Given the following conversation scenario:\n\n"+dial+"\n\n"
    prompt_norm_zipped, fname = [], fname.replace(".json","")
    results[fname] = {"dial":dial,"Qs":[],"norms":[]}
    if len(culture2) > 1:
        for culture1, culture2 in [(culture1,culture2),(culture2,culture1)]:
            prompt_norm_txt1 = prompt+"The discussion topic is about "+indcs+"\n\n"
            prompt_norm_txt1 += "What are some social norms " + \
                "that applies to "+culture1+" culture but not "+culture2+" culture? List them in detail, over separate lines."
            indcs = indcs[:-1] if indcs[-1]=="." else indcs
            prompt_norm_txt2 = prompt + "What are some social norms in discussion about " + indcs + \
                " that applies to "+culture1+" culture but not "+culture2+" culture? List them in detail, over separate lines."
            prompt_norm_zipped.extend([("A "+culture1,prompt_norm_txt1),("B "+culture1,prompt_norm_txt2)])
    else:
        prompt_norm_txt = prompt + "The discussion topic is about "+indcs+"\n\n" 
        prompt_norm_txt += "List a set of social norms unique in "+culture1+" culture:"
        fname = fname.replace(".json","")
        prompt_norm_zipped = [("C "+culture,prompt_norm_txt)]
    for (promptID, prompt_norm_txt) in prompt_norm_zipped:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt = prompt_norm_txt,
            temperature=0.7,
            max_tokens=256,
            top_p=1
        )
        norms = response["choices"][0]["text"].split("\n")
        norms = [x[1:] if x[0]=="-" else x[3:] if x[1:2]==": " else \
                 x[3:] if x[1:2]==") " else x for x in norms if len(x)>0]
        results[fname]["Qs"].extend([promptID]*len(norms))
        results[fname]["norms"].extend(norms)
    with open("data/prompt/norms_GPT3_wIndc/prompted_results_all.json", "w") as f:
        json.dump(results, f)
    
