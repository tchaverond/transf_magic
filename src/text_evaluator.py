import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
from transformers import pipeline
import evaluate


TEST_SIZE = 0.3


def prepare_into_distilbert_dataset():

    with open("data/curated/real_dataset.txt", 'r') as fin:
        real_cards = fin.read().splitlines()
    with open("data/curated/fake_dataset.txt", 'r') as fin:
        fake_cards = fin.read().splitlines()

    real_cards = [{'label': 1, 'text': t} for t in real_cards]
    fake_cards = [{'label': 0, 'text': t} for t in fake_cards]
    
    rng = np.random.default_rng(seed=42)
    rng.shuffle(real_cards)
    rng.shuffle(fake_cards)
    
    real_cards = {'train': real_cards[:int((1-TEST_SIZE)*len(real_cards))],
                  'test': real_cards[int((1-TEST_SIZE)*len(real_cards)):]}
    fake_cards = {'train': fake_cards[:int((1-TEST_SIZE)*len(fake_cards))],
                  'test': fake_cards[int((1-TEST_SIZE)*len(fake_cards)):]}
    all_cards = {'train': real_cards['train'] + fake_cards['train'],
                 'test': real_cards['test'] + fake_cards['test']}
    rng.shuffle(all_cards['train'])
    rng.shuffle(all_cards['test'])
    
    return all_cards


def compute_metrics(eval_pred):

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

dataset = prepare_into_distilbert_dataset()
tokenized = {}
for key, value in dataset.items():
    tokenized[key] = []
    for v in value:
        tokenized_card = tokenizer(v['text'])
        tokenized_card.update({'label': v['label']})
        tokenized[key].append(tokenized_card)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
id2label = {0: 'fake', 1: 'real'}
label2id = {'fake': 0, 'real': 1}

evaluator = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=2, 
    id2label=id2label, label2id=label2id)
training_args = TrainingArguments(output_dir='tmp_trainer')
accuracy = evaluate.load("accuracy")

trainer = Trainer(model=evaluator, args=training_args, train_dataset=tokenized['train'], 
                  eval_dataset=tokenized['test'], tokenizer=tokenizer,
                  data_collator=data_collator, compute_metrics=compute_metrics)
trainer.train()
trainer.push_to_hub("exerah-transf_magic_evaluator_v0_17-07")   # connect via Anaconda console

# retrieval : Note: it's using the name given in output_dir of TrainingArguments
tokenizer = AutoTokenizer.from_pretrained("Exerah/tmp_trainer")
model = AutoModelForSequenceClassification.from_pretrained("Exerah/tmp_trainer")

# use older cards and more difficult fakes for testing?
evaluator = pipeline('text-classification', model="Exerah/tmp_trainer")
result = [evaluator(t['text'])[0] for t in tqdm(dataset['test'])]
comparison = pd.DataFrame(result)
comparison['reference'] = [id2label[t['label']] for t in dataset['test']]
