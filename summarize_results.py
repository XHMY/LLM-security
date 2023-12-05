import json
import pandas as pd
from glob import glob

summary = []

for summary_file in glob("logs/*/*_summary.json"):
    if "draft" in summary_file:
        continue
    with open(summary_file, 'r') as fd:
        data = json.load(fd)

    config_name = summary_file.split("/")[1]
    model_name = summary_file.split("/")[-1].split("_")[2]
    
    for i in data["pass_rates"]:
        summary.append({"config_name": config_name, "model_name": model_name, "combine": i[0], "ASR": i[1]})
    
df = pd.DataFrame(summary)
print(df)
df.to_csv("Summary.csv", index=False)