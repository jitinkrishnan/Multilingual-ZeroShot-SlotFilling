import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# optimizer from hugging face transformerss
device = torch.device("cuda")
cross_entropy  = nn.CrossEntropyLoss()

# function to train the model
def train(model, train_dataloader, alpha, beta, optimizer, schedular, MAX_LEN):
  
    model.train()

    total_loss, total_accuracy = 0, 0
  
    # empty list to save model predictions
    total_preds=[]
  
    # iterate over batches
    steps = []
    batches = []

    for step,batch in enumerate(train_dataloader):
        steps.append(step)
        batches.append(batch)

    for index in range(len(steps)):
        step = steps[index]
        batch = batches[index]
    
        # progress update after every 50 batches.
        #if step % 50 == 0 and not step == 0:
            #print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        # push the batch to gpu
        batch = [r.to(device) for r in batch]
        #batch = [r for r in batch]

        sent_id, mask, slot_labels, labels = batch

        # clear previously calculated gradients 
        model.zero_grad()       

        # get model predictions for the current batch
        predsA, slots = model(sent_id, mask, labels)

        # INTENT
        # compute the loss between actual and predicted values
        lossA = alpha*predsA #cross_entropy(predsA, labels)
        total_loss = total_loss + lossA.item()
        # backward pass to calculate the gradients
        lossA.backward(retain_graph=True)

        # SLOT
        preds = torch.reshape(slots,(slots.shape[0]*MAX_LEN,slots.shape[2]))
        slot_labels = torch.reshape(slot_labels, (slot_labels.shape[0]*MAX_LEN,))
        loss = beta*cross_entropy(preds, slot_labels)
        total_loss = total_loss + loss.item()
        loss.backward()
        
        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()
        schedular.step()

        # model predictions are stored on GPU. So, push it to CPU
        predsA=predsA.detach().cpu().numpy()
        slots=slots.detach().cpu().numpy()

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    #returns the loss and predictions
    return model, avg_loss, total_preds


# function for evaluating the model
def evaluate(model, val_dataloader, alpha, beta, optimizer, schedular, MAX_LEN):
  
    print("\nEvaluating...")

    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    for step,batch in enumerate(val_dataloader):
        # push the batch to gpu
        batch = [t.to(device) for t in batch]
        #batch = [t for t in batch]

        sent_id, mask, slot_labels, labels = batch

        # deactivate autograd
        with torch.no_grad():
            # model predictions
            predsA, slots = model(sent_id, mask)

            # INTENT
            # compute the validation loss between actual and predicted values
            lossA = alpha*cross_entropy(predsA,labels)
            total_loss = total_loss + lossA.item()

            # SLOT
            preds = torch.reshape(slots,(slots.shape[0]*MAX_LEN,slots.shape[2]))
            slot_labels = torch.reshape(slot_labels, (slot_labels.shape[0]*MAX_LEN,))
            loss = beta*cross_entropy(preds, slot_labels)
            total_loss = total_loss + loss.item()
                
            predsA = predsA.detach().cpu().numpy()
            slots = slots.detach().cpu().numpy()

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader) 

    return model, avg_loss, total_preds
