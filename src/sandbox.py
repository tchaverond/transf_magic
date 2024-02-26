# -__-__-__-__-   Small exploration to identify cards for dataset   -__-__-__-__- #

import json
import pandas as pd

scryfall_file = 'data/real/oracle-cards-20240223100133.json'
with open(scryfall_file, 'r') as fin:
    scryfall_raw = json.load(fin)

pd.Series(x['layout'] for x in scryfall_raw).value_counts()
seen_layouts = []
samples = []
for x in scryfall_raw:
    if x['layout'] not in seen_layouts:
        samples.append(x)
        seen_layouts.append(x['layout'])

