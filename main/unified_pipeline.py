from convokit import Corpus, download
import json, os
import openai
from datetime import datetime


def load_USChina_raw():
    return

def load_LDC_raw():
    return

def load_CGA_raw():
    corpus = Corpus(filename=download("conversations-gone-awry-corpus"))
    corpus.print_summary_stats()
    list_of_conv = []
    for convo in corpus.iter_conversations():
        conv, speakers = [], []
        for utterance in convo.iter_utterances():
            if utterance.text.lstrip()[:2] != "==":
                conv.append("Editor "+utterance.speaker.id+": "+utterance.text)
                speakers.append(utterance.speaker.id)
        list_of_conv.append((convo.id, speakers, conv))
    return list_of_conv

def chunkify_dialogues(lns):
    chunks = []
    for i in range(int(len(lns)/5)):
        chunk = lns[5*i:5*(i+1)]
        chunks.append(chunk)
    if int(len(lns)/5)==0:
        chunks = [lns]
    return chunks

def prompt_wSchema(context, schema, questions):
    questions = [x for x in questions if "social" in x][0:2]
    Q_tracker, responses = [], []
    context_header = "Given the following conversation scenario between Wikipedia editors:"
    for question in questions:
        print(datetime.now())
        prompt = context_header+"\n\n"+context
        prompt += "\n\n"+schema
        prompt += "\n\n"+question
        response = openai.Completion.create(
          engine="text-davinci-002",
          prompt = prompt,
          temperature=0.7,
          max_tokens=256
        )
        response = response["choices"][0]["text"].lstrip().split("\n")
        response = [x[1:].lstrip() if x[0]=="-" else x[3:] if x[1:3]==") " \
                     else x for x in response if len(x)>0]
        responses.append(response)
        Q_tracker.append((question, len(response)))
        print(datetime.now())
        quit()
    return Q_tracker, responses

def prompt_grounding(context, schema, speakers):
    #questions = [x for x in questions if "social" in x][0:2]
    speakers = [x for x in speakers if x in context]
    Q_tracker, responses = [], []
    context_header = "Given the following conversation scenario between Wikipedia editors:"
    norms = ["One should always try to be polite and respectful to one's peers.",\
            "One should always try to be truthful and honest in one's reviews."]
    result = []
    for norm in norms: #questions:
        for speaker in speakers:
            prompt = context_header+"\n\n"+context
            prompt += "\n\nAnd given the social norm: "+norm+"\n\n"
            prompt += "Explain whether what's spoken by Editor "+str(speaker)+" in the conversation " + \
                          '<"entails", "is irrelevant with", or "contradicts"> the social norm and why.'
            prompt += " Point to specific instances or word usages."
            #print(prompt)
            response = openai.Completion.create(
              engine="text-davinci-002",
              prompt = prompt,
              temperature=0.7,
              logprobs=5,
              max_tokens=256,
              top_p=1
            )
            response = response["choices"][0]
            response_txt = response["text"].lstrip()#.split("\n")
            #response_txt = [x[1:].lstrip() if x[0]=="-" else x[3:] if x[1:3]==") " \
            #         else x for x in response_txt if len(x)>0]
            result.append((context,speaker,norm,response_txt,\
                          response["logprobs"]["tokens"],response["logprobs"]["top_logprobs"]))
            #response = response["choices"][0]["text"].lstrip().split("\n")
            #responses.append(response)
            #Q_tracker.append(question, len(response))
            print(datetime.now())
    return result #Q_tracker, responses

def norm_prompting(list_of_conv, schema, questions, save_fname, mode="discovery"):
    results = {}
    if os.path.exists(save_fname):
        with open(save_fname, "r") as f:
            results = json.load(f)
    for docID, speakers, doc in list_of_conv:
        if docID in results:
            continue
        print(docID, len(results))
        dial_chunks = chunkify_dialogues(doc)
        if mode == "discovery":
            results[docID] = {"dial":[],"Qs":[],"norms":[]}
        else:
            results[docID] = [] #{"dial":[],"norm":[],"label":[],"token":[]}
        for dial_chunk in dial_chunks:
            context = "\n".join(dial_chunk).replace("\n\n","\n").lstrip()
            if mode == "discovery":
                Q_tracker, norms = prompt_wSchema(context, schema, questions)
                results[docID]["dial"].extend(context)
                results[docID]["Qs"].extend(Q_tracker)
                results[docID]["norms"].extend(norms)
            else:
                result = prompt_grounding(context, schema, speakers)
                results[docID].extend(result)
        with open(save_fname, "w") as f:
            json.dump(results, f)
        if len(results)>100:
            return
    return

openai.organization = os.getenv("OPENAI_ORG")
openai.api_key = os.getenv("OPENAI_API_KEY")

with open("data/tmpl/prompt_norm_discovery.txt", "r") as f:
    schema, questions = f.read().split("\n\nQuestion:\n")
schema, questions = schema.replace("Schema:\n",""), questions.split("\n")

list_of_conv = load_CGA_raw()
save_fname = "data/norm_grounding/CGA.json"
norm_prompting(list_of_conv, schema, questions, save_fname, "grounding")

with open(save_fname, "r") as f:
    data = json.load(f)

#for k, v in data.items():
#    print(k)
#    for d in v:
#        #print(d[0])
        #print(d[2])
#        print(d[1],d[3])
#        quit()
