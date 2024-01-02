import os
import json
import openai

openai.organization = "org-fGi4Hgnq7sCektnzHAWEHeSi"
openai.api_key = os.getenv("OPENAI_API_KEY")

with open("data/prompt/few_shot_train_norm_violation_identification.txt", "r") as f:
    few_shot_train_txt = f.read()

data_dir = "data/dialchunk"
for fname in os.listdir(data_dir):
    if os.path.exists("data/prompt/indicators/"+fname):
        continue
    with open(data_dir+"/"+fname,"r") as f:
        txt = f.read()
    print("~~~")
    # Zero-Shot prompting of the scenario topic
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt = txt+"\nSummarize in a few words what the people are talking about or doing:",
      temperature=0,
      max_tokens=60,
      top_p=1
    )
    with open("data/prompt/indicators/"+fname.replace(".txt",".json"), "w") as f:
        json.dump(response, f)
    with open("data/prompt/indicators/"+fname, "w") as f:
        f.write(response["choices"][0]["text"].replace("\n\n",""))
    # Few-shot prompting of norm violation signals
    #few_shot_train_txt = ""
    #response = openai.Completion.create(
    #  engine="text-davinci-002",
    #  prompt = few_shot_train_txt+txt+"\nIdentify parts in the dialogue that point to something rude, stupid, awkward, or norm-violating (if any):",
    #  temperature=0,
    #  max_tokens=60,
    #  top_p=1
    #)
    #with open("data/prompt/indicators/"+fname.replace(".txt","_explicit_violation_signal.json"), "w") as f:
    #    json.dump(response, f)
    #with open("data/prompt/indicators/"+fname.replace(".txt","_explicit_violation_signal.txt"), "w") as f:
    #    f.write(response["choices"][0]["text"].replace("\n\n",""))
    #quit()

