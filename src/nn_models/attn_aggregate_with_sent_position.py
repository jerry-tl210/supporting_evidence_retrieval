import torch
import torch.nn as nn
from transformers import BertModel
from .attn_aggregate import attention
from ..std import *

logger = logging.getLogger(__name__)


class AttnAggregateSentPosModel(nn.Module):
    def __init__(self, number_of_sentence, trained_baseline_model=None):
        super(AttnAggregateSentPosModel, self).__init__()
        self.number_of_sentence = number_of_sentence
        if trained_baseline_model:
            self.bert = trained_baseline_model.bert
            self.sp_linear = trained_baseline_model.linear
        else:
            self.bert = BertModel.from_pretrained('bert-base-chinese')
            self.sp_linear = nn.Linear(768, 1)

        self.current_sentence_transform = nn.Linear(768, 768, bias=False)
        self.target_sentence_transform = nn.Linear(768, 768, bias=False)
        self.sent_position_embedder = nn.Embedding(number_of_sentence, 768)

        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.attn_linear_c = nn.Linear(1536, 1536)
        self.attn_linear_o = nn.Linear(1536, 1, bias=False)

    def forward_nn(self, batch):
        batch_size = batch['input_ids'].shape[0]
        max_sentence_length = batch['input_ids'].shape[2]
        input_ids = batch['input_ids'].view(-1, max_sentence_length)
        token_type_ids = batch['token_type_ids'].view(-1, max_sentence_length)
        attention_mask = batch['attention_mask'].view(-1, max_sentence_length)
        sentence_mask = batch['sentence_mask'].type(torch.float)  # (batch, 3)
        hidden_state, pooler_output = self.bert(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                token_type_ids=token_type_ids)
        pooler_output = self.dropout(pooler_output)
        pooler_output = pooler_output.view(batch_size, -1, 768)  # (batch, 3, 768)

        # Aggregate
        current_sentence = pooler_output[:, self.number_of_sentence // 2, :].unsqueeze(1)  # (batch, 1, 768)
        current_sentence = self.current_sentence_transform(current_sentence)  # (batch, 1, 768)
        target_sentence = self.target_sentence_transform(pooler_output)  # (batch, 3, 768)
        sent_position_embedding = self.sent_position_embedder(batch['sentence_position']) # (batch, 3, 768)
        target_sentence = target_sentence + sent_position_embedding

        aggregated_sentence, weight = \
            attention(current=current_sentence, target=target_sentence, value=pooler_output, mask=sentence_mask,
                      dropout=None)
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
        predict_label = torch.where(score > threshold, torch.ones(len(score), 1), torch.zeros(len(score), 1))
        predict_label = predict_label.numpy().astype(int).tolist()
        return predict_label, score.numpy().tolist(), weight.cpu().numpy().tolist()
