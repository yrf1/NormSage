import os, re, json


src_dir = "data/dialraw"
tgt_dir = "data/dialchunk/"

src_dir = "data/CCU/LDC2022E11_CCU_TA1_Mandarin_Chinese_Development_Source_Data_R1/data/txt"
tgt_dir = "data/CCU/LDC2022E11_CCU_TA1_Mandarin_Chinese_Development_Source_Data_R1/data/dialchunk/"

for fname in os.listdir(src_dir):
    if ".py" in fname or os.path.exists(tgt_dir+fname.replace(".txt","_0.json")):
        continue
    with open(src_dir+"/"+fname,"r") as f:
        data = f.read().lstrip().split("\n")
    data = [x for x in data if x!="--" and x!=""]
    i, chunk_idx = 0, 0
    while i < len(data):
        j, txt_dial, txt_dial_indc = 0, "", ""
        while j < 5 and i < len(data):
            if (data[i][0]=="[" and "]" in data[i]) or \
                    (data[i][0]=="(" and ")" in data[i]):
                txt_dial_indc += data[i] + "\n"
            else:
                j += 1
                txt_dial += "".join(re.split("\(|\)|\[|\]", data[i])[::2]) + "\n"
                txt_dial_indc += data[i] + "\n"
            i += 1
        with open(tgt_dir+fname.replace(".txt","_"+str(chunk_idx)+".json"),"w") as f:
            json.dump({"txt":txt_dial[:-1],"txt+scene":txt_dial_indc[:-1]},f)
        chunk_idx += 1
# TODOs: Write out all the text file snippets (5 dialogue exchange for probing, 15 dialogue exchange for context)
# TODOs: Split out the video as well (for that full 15)
