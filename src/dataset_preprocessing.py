import glob
import json
import pandas as pd
from tqdm import tqdm

DIFFICULTIES = ["easy", "medium"]
RAW_REAL_PATH = "data/raw/real/*.json"
RAW_FAKE_PATHS = ["data/raw/fake/" + diff + "*.txt" for diff in DIFFICULTIES]
SCRYFALL_COLS = ['id', 'name', 'released_at', 'mana_cost', 'type_line', 'oracle_text']
SCRYFALL_EXTRA_COLS = ['power', 'toughness']
SCRYFALL_ALLOWED_LAYOUTS = ['normal', 'saga', 'prototype', 'mutate', 'leveler', 'case']
# TODO: handle that
SCRYFALL_DOABLE_LAYOUTS = ['transform', 'split', 'adventure', 'flip', 'modal_dfc']


def extract_info_from_scryfall_json(scryfall_json):
    
    if scryfall_json['layout'] not in SCRYFALL_ALLOWED_LAYOUTS:
        return
    # only use most recent cards (from 'Dark Ascension', similar to pioneer-legal set)
    # other option: use '2003' to match cards legal in modern (can have weird abilities and power levels)
    if scryfall_json['released_at'] < '2012':
        return
    # avoid cards outside of standard sets (e.g. unsets), that can have weird abilities
    if scryfall_json['legalities']['commander'] == 'not_legal':
        return
    info = {c: scryfall_json[c] for c in SCRYFALL_COLS}
    try:
        info.update({c: scryfall_json[c] for c in SCRYFALL_EXTRA_COLS})
    except KeyError:
        pass
    # TODO: put this 'loyalty' in global config
    try:
        info['loyalty'] = scryfall_json['loyalty']
    except KeyError:
        pass
    return info


def parse_scryfall_into_text(sf_card):
    
    oracle = sf_card['oracle_text']
    oracle = oracle.replace('\n', ' ')
    oracle = oracle.replace('•', '=')   # choices
    oracle = oracle.replace(sf_card['name'], '~')
    if 'power' in sf_card.keys():
        creature_stats = '(' + sf_card['power'] + '/' + sf_card['toughness'] + ')'
        type_and_stats = sf_card['type_line'] + creature_stats
    elif 'loyalty' in sf_card.keys():
        type_and_stats = sf_card['type_line'] + '((' + sf_card['loyalty'] + '))'
    else:
        type_and_stats = sf_card['type_line']
    card_text = ".\\n".join([sf_card['mana_cost'], type_and_stats, oracle])        
    return card_text


def parse_minmaxir_fake_into_text(fake_card):
    
    parts = fake_card.split('\n')
    if len(parts) < 3:
        return
    cardname = parts[0].split('{')[0].strip()
    mana_cost = parts[0][len(cardname):].split('(')[0].strip()
    if (not mana_cost.startswith('{')) or (not mana_cost.endswith('}')):
        print(fake_card)
        mana_cost = ''
    type_line = parts[1].replace('~', '—')
    oracle = ' '.join(parts[2:])
    if cardname != '':
        oracle = oracle.replace(cardname, '~')
    card_text = ".\\n".join([mana_cost, type_line, oracle])
    return card_text


def run():
    
    # -__-__-   dataset of real cards   -__-__- #
    scryfall_file = glob.glob(RAW_REAL_PATH)[0]
    with open(scryfall_file, 'r') as fin:
        scryfall_raw = json.load(fin)

    # 25/02: 17841/30948 cards (57.6%)
    scryfall = [s for scryfall_json in scryfall_raw
                if (s := extract_info_from_scryfall_json(scryfall_json)) is not None]
    real_cards_text = [parse_scryfall_into_text(s) for s in scryfall]
    
    # -__-__-   dataset of fake cards   -__-__- #
    minmaxir_gpt2 = sum([glob.glob(path) for path in RAW_FAKE_PATHS], [])
    all_fakes = []
    for file in minmaxir_gpt2:
        with open(file, 'r') as fin:
            all_fakes.extend(fin.read().split('\n~~~~~~~~\n'))
    fake_cards_text = [s for fake_text in all_fakes
                       if (s := parse_minmaxir_fake_into_text(fake_text)) is not None]
    
    # -__-__-   write to disk   -__-__- #
    with open("data/curated/real_dataset.txt", 'w') as fout:
        fout.write("\n".join(real_cards_text))
    with open("data/curated/fake_dataset.txt", 'w') as fout:
        fout.write("\n".join(fake_cards_text))
