import math
import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F
from ..std import *

logger = logging.getLogger(__name__)


def attention(current, target, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = current.size(-1)
    scores = torch.matmul(current, target.transpose(-2, -1)) / math.sqrt(d_k)  # batch,1, 3
    scores = scores.squeeze(1)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_weight = F.softmax(scores, dim=-1)
    p_attn = attn_weight.unsqueeze(1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value).squeeze(1), attn_weight


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

        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.attn_linear_c = nn.Linear(1536, 1536)
        self.attn_linear_o = nn.Linear(1536, 1, bias=False)

    def forward_nn(self, batch):
        batch_size = batch['input_ids'].shape[0]
        max_sentence_length = batch['input_ids'].shape[2]

        input_ids = batch['input_ids'].view(-1, max_sentence_length)
        token_type_ids = batch['token_type_ids'].view(-1, max_sentence_length)
        attention_mask = batch['attention_mask'].view(-1, max_sentence_length)
        sentence_mask = batch['sentence_mask'].type(torch.float) # (batch, 3)
        hidden_state, pooler_output = self.bert(input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         token_type_ids=token_type_ids)
        pooler_output = self.dropout(pooler_output)
        pooler_output = pooler_output.view(batch_size, -1, 768)  # (batch, 3, 768)

        # Aggregate
        current_sentence = pooler_output[:, self.number_of_sentence // 2, :].unsqueeze(1)  # (batch, 1, 768)
        current_sentence = self.current_sentence_transform(current_sentence) # (batch, 1, 768)
        target_sentence = self.target_sentence_transform(pooler_output) # (batch, 3, 768)
        aggregated_sentence, weight = \
            attention(current=current_sentence, target=target_sentence, value=pooler_output, mask=sentence_mask, dropout=None)

        # concatenated = torch.cat((current_sentence, target_sentence), dim=-1)  # (batch, 3, 768*2)
        # concatenated = self.attn_linear_c(concatenated).tanh()
        # weight = self.attn_linear_o(concatenated) # (batch, 3, 1)
        # weight = weight + (1.0 - sentence_mask) * -10000
        # weight = F.softmax(weight, dim=1)
        # aggregated_sentence = torch.matmul(weight.transpose(1, 2), target_sentence)  # (batch, 1, 768)
        # aggregated_sentence = aggregated_sentence.squeeze(1)  # (batch, 768)
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
        return predict_label, score.numpy().tolist(), weight.cpu().numpy().tolist()