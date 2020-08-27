import torch
import torch.nn as nn
from transformers import BertModel, BertForNextSentencePrediction
import copy
import torch.nn.functional as F
from ..std import *

logger = logging.getLogger(__name__)

class MultiBERTsModel(nn.Module):

    def __init__(self, number_of_sentence, adjust_weight, trained_baseline_model=None, transform=True):
        super(MultiBERTsModel, self).__init__()
        self.number_of_sentence = number_of_sentence
        self.adjust_weight = adjust_weight
        self.bertNSP = BertForNextSentencePrediction.from_pretrained('bert-base-chinese')
        self.softmax = nn.Softmax(dim=1)
        #self.linear = nn.Linear(768 * self.number_of_sentence, 1)
        #self.bert = BertModel.from_pretrained('bert-base-chinese')
        
        if trained_baseline_model:
            self.bert = trained_baseline_model.bert
            self.sp_linear = trained_baseline_model.linear
        else:
            self.bert = BertModel.from_pretrained('bert-base-chinese')
            self.sp_linear = nn.Linear(768, 1)
        
        if transform:
            self.transform = nn.Linear(768, 768, bias=False)
        
    def forward_nn(self, batch):
        batch_size = batch['input_ids'].shape[0]
        max_sentence_length = batch['input_ids'].shape[2]

        # BERT input
        input_ids = batch['input_ids'].view(-1, max_sentence_length)
        token_type_ids = batch['token_type_ids'].view(-1, max_sentence_length)
        attention_mask = batch['attention_mask'].view(-1, max_sentence_length)
    
        hidden_state, pooler_output = self.bert(input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         token_type_ids=token_type_ids)

        
        
        # BERT NSP input
        nsp_input_ids = batch['nsp_input_ids'].view(-1, max_sentence_length)
        
        nsp_token_type_ids = batch['nsp_token_type_ids'].view(-1, max_sentence_length)
        nsp_attention_mask = batch['nsp_attention_mask'].view(-1, max_sentence_length)
        
        sentence_mask = batch['sentence_mask'].type(torch.float)
        sr_scores = self.bertNSP(input_ids=nsp_input_ids, 
                           attention_mask=nsp_attention_mask,
                           token_type_ids=nsp_token_type_ids)
        
        sr_scores = sr_scores[0].view(batch_size, -1, 2)
        
        # Aggregate
        pooler_output = pooler_output.view(batch_size, -1, 768)  # (batch, 3, 768)
        
        
        if self.adjust_weight:
            if hasattr(self, 'transform'):
                pooler_output = self.transform(pooler_output) # (batch, 3, 768) 
            sr_scores = sr_scores + (1.0 - sentence_mask) * -10000
            weight = self.softmax(sr_scores[:, :, 0]).unsqueeze(1)
        else:
            weight = torch.tensor([[[0.0], [1.0], [0.0]]], device=device)
        
        #print(weight)
        aggregated_sentence = torch.matmul(weight, pooler_output)  # (batch, 1, 768)
        aggregated_sentence = aggregated_sentence.squeeze(1)  # (batch, 768)
        
        #pooler_output = pooler_output.view(batch_size, -1, 768) # (batch, 3, 768)
        #pooler_output = pooler_output.reshape(batch_size, 768 * self.number_of_sentence) # (batch, 3*768)
        #final_output = self.sp_linear(pooler_output) # (batch, 1)
        final_output = self.sp_linear(aggregated_sentence)  # (batch, 1)
        
        return final_output

    def forward(self, batch):

        output = self.forward_nn(batch)
        labels = batch['label'].type(torch.float)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(output, labels)
        return loss
    
    def predict(self, batch, threshold=0.5):
        output = self.forward_nn(batch)
        score = torch.sigmoid(output).cpu()
        print(score)
        #highest_score = max(score[:, 0]).item()

        predict_label = torch.where(score > threshold, torch.ones(len(score),1), torch.zeros(len(score), 1))
        #else:    
            #predict_label = torch.where(score == highest_score, torch.ones(len(score),1), torch.zeros(len(score), 1))
        predict_label = predict_label.numpy().astype(int).tolist()
        return predict_label

"""
    def _predict(self, batch):

        with torch.no_grad():
            output, att_weight = self.forward_nn(batch)
            scores = torch.sigmoid(output)
            scores = scores.cpu().numpy().tolist()

        return scores

    def predict_fgc(self, batch, threshold=0.5):
        scores = self._predict(batch)
        max_i = 0
        max_score = 0
        sp = []

        for i, score in enumerate(scores[0]):

            if score > max_score:
                max_i = i
                max_score = score
            if score >= threshold:
                sp.append(i)

        # This is to ensure there's no empty supporting evidences
        if not sp:
            sp.append(max_i)
        return {'sp': sp, 'sp_scores': scores}
"""