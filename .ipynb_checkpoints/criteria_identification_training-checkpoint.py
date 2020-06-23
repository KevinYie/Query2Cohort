from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import json
import re
import sys

import torch
from torch import nn

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import transformers
from transformers import BertForTokenClassification, AdamW

from tqdm import tqdm, trange

from transformers import get_linear_schedule_with_warmup

from seqeval.metrics import f1_score, accuracy_score


"""
Creates labels for each token for inclusion and exclusion criteria

CSV must have query and cohort columns with each entry in cohort structured as a dictionary with two keys: inclusion and exclusion. Values for thees keys must be lists
    - e.g. {"inclusion": ["example a", "example b"], "exclusion": ["exclusion a"]}

Parameters
----------
document: Path to csv with "query" and "cohort"


Returns
-------
A csv containing original columns with adidtion of inclusion, exclusion, and label columns.

Label example:
    - query = "undergoing routine antenatal care but don't have adverse effect, caused by correct medicinal substance properly administered"
    - labels = "Neither, Neither, include, include, include, Neither, Neither, Neither, Neither, Neither, exclude, exclude, Neither, exclude, exclude, exclude, exclude, Neither, Neither, include"

Labels are at the level of tokens created by the base BertTokenizer

"""

document = sys.argv[1]

# Read in text, create inclusion and exclusion columns, and clean
df = pd.read_csv(str(document))
df = df.reset_index()
df.columns = ["query", "cohort", "intent"]
df["cohort"] = df["cohort"].apply(json.loads) 

df["inclusion"] = ["None"]*len(df)
df["exclusion"] = ["None"]*len(df)

cohort = df["cohort"]
for x in range(len(cohort)):
    df["inclusion"][x] = cohort[x]["inclusion"]
    df["exclusion"][x] = cohort[x]["exclusion"]
    

def clean_text(x):
    x = re.sub("-", "", x)
    x = re.sub("\(", " ", x)
    x = re.sub("\)", " ", x)
    return x
    
df["query"] = df["query"].apply(clean_text)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased',do_lower_case=True)

# Creating labels for each token based on exclusion and inclusion criteria
final_labels = []

for index, row in df.iterrows():
    
    tokenized_query = tokenizer.tokenize(row["query"])
    labels = ["Neither"]*len(tokenized_query)
    tokenized_inclusion = [tokenizer.tokenize(x) for x in row["inclusion"]]
    tokenized_exclusion = [tokenizer.tokenize(x) for x in row["exclusion"]]
    
    for criteria in tokenized_inclusion:
        for token in range(int(len(tokenized_query)-len(criteria))+1):
            if tokenized_query[token:token+len(criteria)] == criteria:
                labels[token:token+len(criteria)] = ["include"] * len(criteria)
                
    for criteria in tokenized_exclusion:
        for token in range(int(len(tokenized_query)-len(criteria))+1):
            if tokenized_query[token:token+len(criteria)] == criteria:
                labels[token:token+len(criteria)] = ["exclude"] * len(criteria)
    
    final_labels.append(labels)

df["labels"] = final_labels

df.to_csv("final.csv", index = False)


"""
Finetuning BERT for Cohort Identification

Parameters
----------
epochs = How many epochs to train over


Returns
-------
Fine-tuned model saved as state dictionary with final validation F1 score and accuracy

"""

# Creating conversion dictionary
tag_values = list(set(df["labels"].values[0]))
tag_values.append("PAD")
tag2idx = {t:i for i, t in enumerate(tag_values)}

# Parameters
device = torch.device("cuda")
MAX_LEN = max([len(x) for x in df["labels"]])
bs = 32

# Creating input data
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
tokenized_texts = [tokenizer.tokenize(query) for query in df["query"]]
labels = list(df["labels"])


# Padding inputs and converting tokens to ids
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", value=0.0,
                          truncating="post", padding="post")

# Padding tags and converting to ids
tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                     dtype="long", truncating="post")

# Creating masks
attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

# Training and validation split
tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                            random_state=42, test_size=0.3)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=42, test_size=0.3)

# Converting all to tensors
tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)

# Dataloader creation
train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

# Model definition and writing to CUDA
model = BertForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=len(tag2idx),
    output_attentions = False,
    output_hidden_states = False
)

model.cuda()

#Setting parameters for fine-tuning
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=3e-5,
    eps=1e-8
)

# Setting training definition
epochs = 6
max_grad_norm = 1.0

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)



    # ========================================
    #               Training Loop
    # ========================================
    
loss_values, validation_loss_values = [], []

for _ in trange(epochs, desc="Epoch"):

# Training

    model.train()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        model.zero_grad()
        
        # forward pass
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        loss.backward()
        total_loss += loss.item()
        
        # Clip the norm of the gradient
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)

        optimizer.step()
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)

    # Store the loss value.
    loss_values.append(avg_train_loss)



# Validation

    # Put the model into evaluation mode
    model.eval()
    # Reset the validation loss for this epoch.
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Not storing gradient for memory 
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)

        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()


        eval_loss += outputs[0].mean().item()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)

    eval_loss = eval_loss / len(valid_dataloader)
    validation_loss_values.append(eval_loss)
    
    pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    valid_tags = [tag_values[l_i] for l in true_labels
                                  for l_i in l if tag_values[l_i] != "PAD"]
print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))

torch.save(model.state_dict(), "Criteria.pth")
