import pandas as pd
import random
import json

df = pd.read_csv("data/MTurk/batches_output_grounding_comparison/norm_grounding_comparison_Batch_4747801_batch_results.csv")
new_df = [("path","dial","norm","verdict1","expl1","verdict2","expl2","verdict3","expl3","verdict4","expl4","expl_order")]

with open("data/TuhinNormGroundExplPredictions.json", "r") as f:
    TuhinGroundings = json.load(f)

with open("data/baseline_T0_ground_expl.json", "r") as f:
    T0Groundings = json.load(f)

for idx, data in df.iterrows():
    path, dial, norm = data["Input.path"], data["Input.dial"], data["Input.norm"]
    # Human and T5
    expl_order = data["Input.expl_order"]
    verdict1, expl1 =  data["Input.verdict1"],data["Input.expl1"]
    verdict2, expl2 = data["Input.verdict2"],data["Input.expl2"]
    #Tuhin
    Tuhin_data = [x for x in TuhinGroundings if x["hypothesis"]==norm][0]
    verdict3, expl3 = Tuhin_data["predicted_label"], Tuhin_data["model_explanation"]
    #T0
    verdict4, expl4 = T0Groundings[path][-1], T0Groundings[path][1]
    #eSNLI
    k1 = {"man":"manual", "auto":"NormSage"}[expl_order.split("_")[0]]
    k2 = {"man":"manual", "auto":"NormSage"}[expl_order.split("_")[-1]]
    cache = {k1:(verdict1,expl1),k2:(verdict2,expl2)}
    cache["tuhin"] = (verdict3, expl3)
    cache["t0"] = (verdict4, expl4)
    random_choices = ["manual","NormSage","tuhin","t0"]
    random.shuffle(random_choices)
    choice1, choice2, choice3, choice4 = random_choices
    #choice1 = random.choices(random_choices)[0]
    verdict_expls = cache[choice1]
    expl_order = [choice1]
    #random_choices = random_choices - [choice1]
    #choice2 = random.choices(random_choices)[0]
    verdict_expls += cache[choice2]
    expl_order.append(choice2)
    #random_choices = random_choices - [choice2]
    #choice3 = random.choices(random_choices)[0]
    verdict_expls += cache[choice3]
    expl_order.append(choice3)
    #random_choices = random_choices - [choice3]
    #choice4 = random.choices(random_choices)[0]
    verdict_expls += cache[choice4]
    expl_order.append(choice4)
    expl_order = "_".join(expl_order)
    new_df.append((path,dial,norm)+verdict_expls+(expl_order,))

new_df = pd.DataFrame(new_df[1:], columns=new_df[0])
new_df.to_csv("data/MTurk/batches_input_grounding_comparison/norm_grounding_comparison_Batch_4747801_batch_results_expanded.csv")
