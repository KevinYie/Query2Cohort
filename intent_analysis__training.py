from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig
from transformers import BertModel, BertForSequenceClassification
from transformers.optimization import AdamW

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from seqeval.metrics import f1_score, accuracy_score
from sklearn.metrics import f1_score as f1

import copy
import sys


"""
Creates labels for each token for inclusion and exclusion criteria

CSV must have query and cohort columns with each entry in cohort structured as a dictionary with two keys: inclusion and exclusion. Values for thees keys must be lists
    - e.g. {"inclusion": ["example a", "example b"], "exclusion": ["exclusion a"]}

Parameters
----------
document: Path to csv with "query" and "cohort"
epochs: Number of epochs to train over



Returns
-------
A csv containing original columns with adidtion of inclusion, exclusion, and label columns.

Label example:
    - query = "undergoing routine antenatal care but don't have adverse effect, caused by correct medicinal substance properly administered"
    - labels = "Neither, Neither, include, include, include, Neither, Neither, Neither, Neither, Neither, exclude, exclude, Neither, exclude, exclude, exclude, exclude, Neither, Neither, include"

Labels are at the level of tokens created by the base BertTokenizer

"""

document = sys.argv[0]
epochs = sys.argv[1]

#Read in text, create inclusion and exclusion columns, and clean
df = pd.read_csv(str(document))
df.reset_index(inplace = True)
df.columns = ["query", "cohort", "intent"]
df["cohort"] = df["cohort"].apply(json.loads) 

# Convert labels to number
label_values = list(set(df["intent"].values))
labels2idx = {t:i for i, t in enumerate(label_values)}
df["intent"] = df["intent"].apply(labels2idx.get)


"""
Finetuning BERT for Intent Analysis

Parameters
----------
epochs = How many epochs to train over


Returns
-------
Fine-tuned model saved as state dictionary with final validation F1 score and accuracy

"""

device = torch.device("cuda")

# Creating train validation split
msk = np.random.rand(len(df)) < 0.7
train = df[msk]
val = df[~msk]

# Creating function to tokenize inputs, create masks, and output data as tensor dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)

def bert_load(data, inputs, labels, max_len):
    '''
    Load in data
    Return BERT's preprocessed inputs including token_id, mask, label
    '''
    token_ids = []
    attention_masks = []
    for row in data[str(inputs)]:
        encoded_dict = tokenizer.encode_plus(row,
                                            add_special_tokens= True, #add [CLS], [SEP]
                                            max_length = max_len,  
                                            pad_to_max_length = True, #pad and truncate
                                            return_attention_mask = True, #construct attention mask
                                            return_tensors = 'pt') #return pytorch tensor
        
        token_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    token_ids = torch.cat(token_ids,dim=0)
    attention_masks = torch.cat(attention_masks,dim=0)
    labels = torch.tensor(data[str(labels)].values)
    data_out = TensorDataset(token_ids, attention_masks, labels)
    return data_out

# Setting max length and creating data and validation loaders
BATCH_SIZE = 32

max_len = max([len(tokenizer.tokenize(query)) for query in df["query"]])
datatrain = bert_load(train, "query", "intent", max_len)   
dataval = bert_load(val, "query", "intent", max_len)

trainloader = DataLoader(datatrain,
                           batch_size=BATCH_SIZE,
                           shuffle=True)

valloader = DataLoader(dataval,
                           batch_size=BATCH_SIZE,
                           shuffle=True)


# Driver function
def trainBERT(model, trainloader, val_loader, num_epoch=20):
    
    # Training steps
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5, eps= 1e-8) 
    
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_f1 = 0.

    all_f1 = []
    all_pred = []
    all_label = []

    for epoch in range(num_epoch):
        model.train()
        #Initialize
        correct = 0
        total = 0
        total_loss = 0
        pred_list = []
        labels_list = []
        f1_scores = []

        for i, (data, mask, labels) in enumerate(trainloader):
            data, mask, labels = data.to(device), mask.to(device), labels.to(device)
            model.zero_grad()

            loss, outputs = model(data, token_type_ids = None,
                                  attention_mask= mask,
                                  labels =labels)

            loss.backward()
            optimizer.step()
            label_cpu = labels.squeeze().to('cpu').numpy()
            pred = outputs.data.max(-1)[1].to('cpu').numpy()
            
            # For accuracy
            total += labels.size(0)
            correct += float(sum((pred ==label_cpu)))
            total_loss += loss.item()
            
            # For F1
            f1_scores.append(f1(list(pred), list(label_cpu), average = "weighted"))
            
            all_f1.append(f1(list(pred), list(label_cpu), average = "weighted"))
            all_pred.append(pred)
            all_label.append(label_cpu)
            
        acc = correct/total
        t_loss = total_loss/total
        train_loss.append(t_loss)
        train_acc.append(acc)
        
        
        # report performance 
        print("Train Loss: {}".format(t_loss))
        print("Train Accuracy: {}".format(acc))
        print("Train F1-Score: {}".format(sum(f1_scores)/len(f1_scores)))
        print()
    
    # Evaluate after every epoch
        #Reset the initialization
        correct = 0
        total = 0
        total_loss = 0
        model.eval()
        
        predictions =[]
        truths= []
        val_f1_scores = []

        with torch.no_grad():
            for i, (data, mask, labels) in enumerate(val_loader):
                data, mask, labels = data.to(device), mask.to(device), labels.to(device)
                model.zero_grad()

                va_loss, outputs = model(data, token_type_ids = None,
                                      attention_mask= mask,
                                      labels =labels)

                label_cpu = labels.squeeze().to('cpu').numpy()
                
                pred = outputs.data.max(-1)[1].to('cpu').numpy()
                total += labels.size(0)
                correct += float(sum((pred == label_cpu)))
                total_loss += va_loss.item()
                
                predictions += list(pred)
                truths += list(label_cpu)
                
                #F1 scores calculation
                val_f1_scores.append(f1(list(pred), list(label_cpu), average = "weighted"))
                       
            v_acc = correct/total
            v_loss = total_loss/total
            val_loss.append(v_loss)
            val_acc.append(v_acc)
            v_f1 = sum(val_f1_scores)/len(val_f1_scores)
            
            print("Validation Loss: {}".format(v_loss))
            print("Validation Accuracy: {}".format(v_acc))
            print("Validation F1-Score: {}".format(v_f1))
            print()
             

# Model Definition
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(labels2idx))
model.cuda()

# Running Driver function
train_loss, train_acc, val_loss, val_acc, val_f1, model, all_f1, all_pred, all_label = trainBERT(model, trainloader, valloader, num_epoch=epochs)

torch.save(model, "Intent.pth")
