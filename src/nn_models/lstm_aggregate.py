import torch
import torch.nn as nn
from transformers import BertModel
import copy
import torch.nn.functional as F
from ..std import *

logger = logging.getLogger(__name__)

class FGC_LSTM_Network(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        
        super(FGC_LSTM_Network, self).__init__()
        self.bert = trained_baseline_model.bert
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.h0 = nn.Parameter(torch.FloatTensor(hidden_size).uniform_(-0.1, 0.1))
        self.c0 = nn.Parameter(torch.FloatTensor(hidden_size).uniform_(-0.1, 0.1))
        self.linear = nn.Linear(hidden_size*2, 1) 
        # (H-state and c_state must be converted into 3d in forward function)
        
    def forward_nn(self, batch):    
        # batch['ids'] = (batch_size*number of sentence, sent_len)
        # batch['segment_ids'] = (batch_size*number of sentence, sent_len)
        # batch['mask_ids'] = (batch_size*number of sentence, sent_len)
        # pooler_output = (batch_size, 768)
        # hidden_state = (batch_size, sent_len, 768)
        # output = (batch_size, 1)
        
        #h0 = torch.zeros(self.num_layers*2, batch['ids'].shape[0], self.hidden_size).to(device)
        #c0 = torch.zeros(self.num_layers*2, batch['ids'].shape[0], self.hidden_size).to(device)
        
        
        batch_size = batch['input_ids'].shape[0]
        max_sentence_length = batch['input_ids'].shape[2]
        
        #h0 = self.h0.expand(batch_size, self.num_layers*2, -1)
        #c0 = self.c0.expand(batch_size, self.num_layers*2, -1)
        input_ids = batch['input_ids'].view(-1, max_sentence_length)
        token_type_ids = batch['token_type_ids'].view(-1, max_sentence_length)
        attention_mask = batch['attention_mask'].view(-1, max_sentence_length)
        sentence_mask = batch['sentence_mask'].type(torch.float)
        
        hidden_state, pooler_output = self.bert(input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         token_type_ids=token_type_ids)  
        
        hidden_state = hidden_state[:,0].view(batch_size, -1, 768)
        
        lstm_output, (hn, cn) = self.lstm(hidden_state)
        
        linear_output = self.linear(lstm_output).squeeze(-1)
        
        return linear_output
    
    def forward(self, batch):
        output = self.forward_nn(batch)
        labels = batch['labels']
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(output, labels)
        return loss
    
    def predict(self, batch, threshold=0.5):
        output = self.forward_nn(batch)
        score = torch.sigmoid(output).cpu()
        predict_label = torch.where(score > threshold, torch.ones(len(score),1), torch.zeros(len(score), 1))
        predict_label = predict_label.numpy().astype(int).tolist()
        return predict_label