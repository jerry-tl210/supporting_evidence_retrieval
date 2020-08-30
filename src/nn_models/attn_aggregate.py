import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F
from ..std import *

logger = logging.getLogger(__name__)


class AttnAggregateModel(nn.Module):
    def __init__(self, number_of_sentence, adjust_weight, trained_baseline_model=None, transform=True):
        super(AttnAggregateModel, self).__init__()
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

        self.weight_dropout = nn.Dropout(0.2)
        self.weightTransform = nn.Linear(1536, 1, bias=False)

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
                current_sentence = self.current_sentence_transform(current_sentence).tanh() # (batch, 1, 768)
                target_sentence = self.target_sentence_transform(pooler_output).tanh() # (batch, 3, 768)
                
            else:
                current_sentence = pooler_output[:, self.number_of_sentence // 2, :].unsqueeze(1)  # (batch, 1, 768)
                target_sentence = pooler_output
                
            current_sentence = current_sentence.expand(-1, self.number_of_sentence, -1)  # (batch, 3, 768)
            concatenated = torch.cat((current_sentence, target_sentence), dim=-1)  # (batch, 3, 768*2)
            concatenated = self.weight_dropout(concatenated) # (batch, 3, 1536)
            weight = self.weightTransform(concatenated) # (batch, 3, 1)
            weight = weight + (1.0 - sentence_mask) * -10000
            weight = F.softmax(weight, dim=1)
        else:            
            weight = torch.tensor([[[0.0], [1.0], [0.0]]], device=device)
            
        aggregated_sentence = torch.matmul(weight.transpose(1, 2), target_sentence)  # (batch, 1, 768)
        aggregated_sentence = aggregated_sentence.squeeze(1)  # (batch, 768)
        logits = self.sp_linear(aggregated_sentence)  # (batch, 1)

        return logits, weight

    def forward(self, batch):
        logits, _ = self.forward_nn(batch)
        labels = batch['label'].type(torch.float)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, labels)
        return loss
    
    def predict(self, batch, threshold=0.5):
        logits, weight = self.forward_nn(batch)
        score = torch.sigmoid(logits).cpu()
        predict_label = torch.where(score > threshold, torch.ones(len(score),1), torch.zeros(len(score), 1))
        predict_label = predict_label.numpy().astype(int).tolist()
        return predict_label, score.numpy().tolist(), weight.numpy().tolist()