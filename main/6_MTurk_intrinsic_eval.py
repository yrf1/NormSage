import pandas as pd


df = pd.read_csv("data/MTurk/batches_output/Batch_354267_batch_results.csv")

for question_keyword in ["insightfulness","diversity"]:
    print(question_keyword)
    a, b, c = [], [], []
    for idx, data in df.iterrows():
        if question_keyword not in data['Input.question_v2']:
            continue
        if data['Input.norm_order'] == "abc":
            a.append(data['Answer.howMuch1'])
            b.append(data['Answer.howMuch2'])
            c.append(data['Answer.howMuch3'])
        if data['Input.norm_order'] == "acb":
            a.append(data['Answer.howMuch1'])
            b.append(data['Answer.howMuch3'])
            c.append(data['Answer.howMuch2'])
        if data['Input.norm_order'] == "bac":
            a.append(data['Answer.howMuch2'])
            b.append(data['Answer.howMuch1'])
            c.append(data['Answer.howMuch3'])
        if data['Input.norm_order'] == "bca":
            a.append(data['Answer.howMuch2'])
            b.append(data['Answer.howMuch3'])
            c.append(data['Answer.howMuch1'])
        if data['Input.norm_order'] == "cab":
            a.append(data['Answer.howMuch3'])
            b.append(data['Answer.howMuch1'])
            c.append(data['Answer.howMuch2'])
        if data['Input.norm_order'] == "cba":
            a.append(data['Answer.howMuch3'])
            b.append(data['Answer.howMuch2'])
            c.append(data['Answer.howMuch1'])
    print(sum(a)/len(a),sum(b)/len(b),sum(c)/len(c))

