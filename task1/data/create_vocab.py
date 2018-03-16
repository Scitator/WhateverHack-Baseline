import pandas as pd
from collections import defaultdict

DATA = "./train_data.csv"
FILE_OUT = "./vocab.txt"
df = pd.read_csv(DATA, names=["image_name", "text", "label"])

counter = defaultdict(lambda: 0)
for i, row in df.iterrows():
    words = row["text"].lower().split()
    for word in words:
        counter[word] += 1

sorted_counter = sorted(counter.items(), key=lambda x: -x[1])

with open(FILE_OUT, "w") as fout:
    for key, value in sorted_counter:
        fout.write(key + "\t" + str(value) + "\n")
