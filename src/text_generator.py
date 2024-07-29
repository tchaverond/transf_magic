import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

TEST_SIZE = 0.3


def prepare_into_distilgpt2_dataset():

    with open("data/curated/real_dataset.txt", 'r') as fin:
        real_cards = fin.read().splitlines()
    
    rng = np.random.default_rng(seed=42)
    rng.shuffle(real_cards)

    real_cards = {'train': real_cards[:int((1-TEST_SIZE)*len(real_cards))],
                  'test': real_cards[int((1-TEST_SIZE)*len(real_cards)):]}

    return real_cards

dataset = prepare_into_distilgpt2_dataset()
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenized = {'train': [tokenizer(sample) for sample in dataset['train']],
             'test': [tokenizer(sample) for sample in dataset['test']]}
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

text_generator = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
training_args = TrainingArguments(output_dir='mtg_card_text_generator', learning_rate=5e-5)

trainer = Trainer(model=text_generator, args=training_args, train_dataset=tokenized['train'],
                  eval_dataset=tokenized['test'], tokenizer=tokenizer, data_collator=data_collator)
trainer.train()
trainer.push_to_hub("exerah-transf_magic_card_generator_v0_23-07")


from transformers import pipeline

generator = pipeline("text-generation", model="Exerah/mtg_card_text_generator")

