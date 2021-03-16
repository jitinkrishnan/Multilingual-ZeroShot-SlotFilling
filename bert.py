import torch
import torch.nn as nn

class mBERT(nn.Module):

    def __init__(self, bert, max_length, slot_classes, num_classes, hidden_layers, drop=0.1):
      
      super(mBERT, self).__init__()

      self.bert = bert
      #self.relu =  nn.ReLU()
      #self.dropout = nn.Dropout(drop)
      self.dropout = nn.Dropout(p=drop, inplace=False)
      self.fcSl = nn.Linear(hidden_layers,slot_classes)

    #define forward pass
    def forward(self, sent_id, mask, labels=None):

      if labels is None:
          out = self.bert(sent_id, 
              token_type_ids=None, 
              attention_mask=mask)
      else:
          out = self.bert(sent_id, 
              token_type_ids=None, 
              attention_mask=mask, 
              labels=labels)

      slots_vec = out.hidden_states[-1]

      slots = self.dropout(slots_vec)
      slots = self.fcSl(slots)

      return out[0], slots