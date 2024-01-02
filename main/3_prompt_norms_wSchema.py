import os
import json
import openai
import pandas as pd


openai.organization = os.getenv("OPENAI_ORG")
openai.api_key = os.getenv("OPENAI_API_KEY")

results = {}

try:
    with open("data/norm_discoveries/NormSage_wSchema_LDC.json", "r") as f:
        results = json.load(f)
except:
    pass

src_dial_dir = "data/dialchunk/"
src_dial_dir = "data/CCU/LDC2022E11_CCU_TA1_Mandarin_Chinese_Development_Source_Data_R1/data/dialchunk/"

for fname in os.listdir(src_dial_dir):
    if fname in results:
        continue
    context_header = "Given the following conversation scenario:"
    with open(src_dial_dir+fname, "r") as f:
        context = json.load(f)["txt"].replace("<b>","").replace("</b>","").replace("</font>","")
        context = context.replace("<font color=#","<").replace("<cf685c>","")
        try:
            context = "\n".join([x if x[0]!="<" else "".join(x.split(">")[1:]) for x in context.split("\n")])
        except:
            pass
    with open("data/tmpl/prompt_norm_discovery.txt", "r") as f:
        schema, questions = f.read().split("\n\nQuestion:\n")
    for question in questions.split("\n")[3:6]:
        prompt = context_header+"\n\n"+context+"\n\n"+schema+"\n\n"+question
        response = openai.Completion.create(
          engine="text-davinci-002",
          prompt = prompt,
          temperature=0.7,
          max_tokens=256
        )
        response = response["choices"][0]["text"].lstrip().split("\n")
        response = [x[1:].lstrip() if x[0]=="-" else x[3:] if x[1:3]==") " else x for x in response if len(x)>0]
        if fname not in results:
            results[fname] = {"dial":context,"Qs":[],"norms":[]}
        results[fname]["Qs"].append((question, len(response)))
        results[fname]["norms"].extend(response)
    with open("data/norm_discoveries/NormSage_wSchema_LDC.json", "w") as f:
        json.dump(results, f)
