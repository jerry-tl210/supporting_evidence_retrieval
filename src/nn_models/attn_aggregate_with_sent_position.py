import torch
import torch.nn as nn
from transformers import BertModel
import copy
import torch.nn.functional as F
from ..std import *

logger = logging.getLogger(__name__)


class AttnAggregateSentPosModel(nn.Module):
    
    def __init__(self, number_of_sentence, adjust_weight, sent_emb_dim=100, trained_baseline_model=None, transform=True):
        super(AttnAggregateSentPosModel, self).__init__()
        logger.info("self.adjust_weight:{}".format(adjust_weight))
        self.number_of_sentence = number_of_sentence
        self.adjust_weight = adjust_weight
        if trained_baseline_model:
            self.bert = trained_baseline_model.bert
            self.sp_linear = trained_baseline_model.linear
        else:
            self.bert = BertModel.from_pretrained('bert-base-chinese')
            self.sp_linear = nn.Linear(768, 1)
        if transform:
            self.current_sentence_transform = nn.Linear(768, 768, bias=False)
            self.target_sentence_transform = nn.Linear(768, 768, bias=False)
            
        self.sent_position_embedder = nn.Embedding(number_of_sentence, embedding_dim=sent_emb_dim)
        self.linearAgg = nn.Linear(1536+sent_emb_dim, 768)
        self.weightTransform = nn.Linear(768, 1)
    
    def forward_nn(self, batch):
        batch_size = batch['input_ids'].shape[0]
        max_sentence_length = batch['input_ids'].shape[2]
        
        input_ids = batch['input_ids'].view(-1, max_sentence_length)
        device = input_ids.device
        token_type_ids = batch['token_type_ids'].view(-1, max_sentence_length)
        attention_mask = batch['attention_mask'].view(-1, max_sentence_length)
        sentence_mask = batch['sentence_mask'].type(torch.float)
        hidden_state, pooler_output = self.bert(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                token_type_ids=token_type_ids)
        
        # Aggregate
        pooler_output = pooler_output.view(batch_size, -1, 768)  # (batch, 3, 768)
        if self.adjust_weight:
            if hasattr(self, 'current_sentence_transform'):
                current_sentence = pooler_output[:, self.number_of_sentence // 2, :].unsqueeze(1)  # (batch, 1, 768)
                current_sentence = self.current_sentence_transform(current_sentence)  # (batch, 1, 768)
                
                target_sentence = self.target_sentence_transform(pooler_output)  # (batch, 3, 768)
            
            else:
                current_sentence = pooler_output[:, self.number_of_sentence // 2, :].unsqueeze(1)  # (batch, 1, 768)
                target_sentence = pooler_output
            
            current_sentence = current_sentence.expand(-1, self.number_of_sentence, -1)  # (batch, 3, 768)
            sent_position_embedding = self.sent_position_embedder(batch['sentence_position']) #(batch, 3, sent_emb_dim)
            concatenated = torch.cat((current_sentence, target_sentence, sent_position_embedding), dim=-1)  # (batch, 3, 768*2)
            weight = self.linearAgg(concatenated)  # (batch, 3, 1536)
            weight = weight.tanh()  # (batch, 3, 1536)
            weight = self.weightTransform(weight)  # (batch, 3, 1)
            weight = weight + (1.0 - sentence_mask) * -10000
            weight = F.softmax(weight, dim=1)
        else:
            weight = torch.tensor([[[0.0], [1.0], [0.0]]], device=device)
        
        aggregated_sentence = torch.matmul(weight.transpose(1, 2), target_sentence)  # (batch, 1, 768)
        aggregated_sentence = aggregated_sentence.squeeze(1)  # (batch, 768)
        
        final_output = self.sp_linear(aggregated_sentence)  # (batch, 1)
        
        return weight, final_output
    
    def forward(self, batch):
        
        weight, output = self.forward_nn(batch)
        labels = batch['label'].type(torch.float)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(output, labels)
        
        return loss
    
    def predict(self, batch, threshold=0.5):
        weight, output = self.forward_nn(batch)
        score = torch.sigmoid(output).cpu()
        highest_score = max(score[:, 0]).item()
        # if highest_score > threshold:
        predict_label = torch.where(score > threshold, torch.ones(len(score), 1), torch.zeros(len(score), 1))
        '''
        else:
            predict_label = torch.where(score == highest_score, torch.ones(len(score),1), torch.zeros(len(score), 1))
        '''
        predict_label = predict_label.numpy().astype(int).tolist()
        return weight, predict_label


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