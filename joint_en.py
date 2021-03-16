import numpy as np
import pandas as pd
import torch, sys
import torch.nn as nn
from sklearn.metrics import classification_report
import transformers
from transformers import get_linear_schedule_with_warmup, AdamW
from transformers import BertForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score
from data_utils import *
from train_utils import *
from bert import *
# optimizer from hugging face transformers
device = torch.device("cuda")

data_folder = sys.argv[1]
code_switch = sys.argv[2]

TRAIN_DOMAIN = 'EN'
SCRATCH_FNAME = "joint_en.pt" # where to store
alpha = 1.0
beta = 1.0
FREEZE = 4

hidden_layers = 768
epochs = 25
delta = 5e-4
lr = 5e-5
patience = 5

TEST_DOMAINS = ['ES', 'DE', 'ZH', 'JA', 'PT', 'FR', 'HI', 'TR']

## DATASETS
TRAIN_TSV = data_folder+"train_"+TRAIN_DOMAIN+".tsv"
VAL_TSV = data_folder+"dev_"+TRAIN_DOMAIN+".tsv"

TEST_TSVs = []

for t in TEST_DOMAINS:
    TEST_TSVs.append(data_folder+"test_"+t+".tsv")

############### CREATE TOKENIZER ########################## 
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")  

############### LOAD DATASET ##########################
if not code_switch:
    train_seq, train_mask, train_slot_y, train_y, sentenceLabel2idx, tag2idx, MAX_LEN = getData4Bert(TRAIN_TSV, tokenizer)
else:
    train_seq, train_mask, train_slot_y, train_y, sentenceLabel2idx, tag2idx, MAX_LEN = getData4BertPickle(data_folder+'train_multi_en.p', tokenizer)
num_slot_classes = len(tag2idx)
num_classes = len(sentenceLabel2idx)

if not code_switch:
    val_seq, val_mask, val_slot_y, val_y, _, _, _  = getData4Bert(VAL_TSV, tokenizer, sentenceLabel2idx, tag2idx, MAX_LEN)
else:
    val_seq, val_mask, val_slot_y, val_y, _, _, _  = getData4BertPickle(data_folder+'dev_multi_en.p', tokenizer, sentenceLabel2idx, tag2idx, MAX_LEN)

TEST_DATA = []
for t in TEST_TSVs:
  td = getData4Bert(t, tokenizer, sentenceLabel2idx, tag2idx, MAX_LEN)
  TEST_DATA.append(td[:4])

train_dataloader = create_loaders(train_seq, train_mask, train_slot_y, train_y)
val_dataloader = create_loaders(val_seq, val_mask, val_slot_y, val_y)


############### LOAD BERT ##########################
bert = BertForSequenceClassification.from_pretrained("bert-base-multilingual-uncased", num_labels = num_classes, output_attentions = False, output_hidden_states = True, return_dict=True)

############### FREEZE ##########################
if FREEZE == 0:
    pass
else:
    for param in bert.bert.embeddings.parameters():
        param.requires_grad = False
    for index in range(len(bert.bert.encoder.layer)):
        if index < FREEZE:
            for param in bert.bert.encoder.layer[index].parameters():
                param.requires_grad = False

############### LOAD BERT ##########################
model = mBERT(bert, MAX_LEN, num_slot_classes, num_classes, hidden_layers)

# push the model to GPU
model = model.to(device)

# define the optimizer
optimizer = AdamW(model.parameters(), lr = lr)          # learning rate

schedular = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_dataloader)*epochs,
)

# set initial loss to infinite
best_valid_loss = float('inf')

# empty lists to store training and validation loss of each epoch
train_losses=[]
valid_losses=[]
best_epoch = 0

#for each epoch
for epoch in range(epochs):
    
    print('\nEpoch {:} / {:}'.format(epoch + 1, epochs), end=" ")
    
    #train model
    model, train_loss, _ = train(model, train_dataloader, alpha, beta, optimizer, schedular, MAX_LEN)
    
    #evaluate model
    model, valid_loss, _ = evaluate(model, val_dataloader, alpha, beta, optimizer, schedular, MAX_LEN)
    
    #save the best model
    if valid_loss < best_valid_loss and epoch > 5:
        best_valid_loss = valid_loss
        #torch.save(model.state_dict(), SCRATCH_FNAME)
        best_epoch = epoch+1
    
    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    #print(f'Training Loss: {train_loss:.5f} Validation Loss: {valid_loss:.5f}')

    if epoch > 5:
        if not is_changing(valid_losses[-1*patience:], delta):
            print("STOPPING - NO CHANGE")
            break

print('BEST EPOCH= ', best_epoch)

############### predict ########################## 

for index in range(len(TEST_DOMAINS)):
    test_seq, test_mask, test_slot_y, test_y = TEST_DATA[index]
    with torch.no_grad():
        preds, slots = model(test_seq.to(device), test_mask.to(device))
        preds = preds.detach().cpu().numpy()
        slots = slots.detach().cpu().numpy()

    preds = np.argmax(preds, axis = 1)
    #print(classification_report(test_y, preds, digits=4))
    acc = accuracy_score(test_y, preds)
    print("Intent Acc.: "+TEST_DOMAINS[index]+": "+str(acc))


    preds = np.reshape(slots,(slots.shape[0]*MAX_LEN,slots.shape[2]))
    slot_labels = np.reshape(test_slot_y, (test_slot_y.shape[0]*MAX_LEN,))
    preds = np.argmax(preds, axis = 1)
    #print(classification_report(slot_labels, preds, digits=4))
    f1 = f1_score(slot_labels, preds, average='micro')
    print("Slot F1: "+TEST_DOMAINS[index]+": "+str(f1))