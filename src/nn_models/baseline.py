import torch
import torch.nn as nn
from transformers import BertModel


class BaselineModel(nn.Module):

    def __init__(self):

        super(BaselineModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.linear = nn.Linear(768, 1)

    def forward(self, batch):
        # batch['ids'] = (batch_size, sent_len)
        # batch['segment_ids'] = (batch_size, sent_len)
        # batch['mask_ids'] = (batch_size, sent_len)
        # pooler_output = (batch_size, 768)
        # output = (batch_size, 1)

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']

        hidden_state, pooler_output = self.bert(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                token_type_ids=token_type_ids)
        
        linear_output = self.linear(pooler_output)

        return linear_output

    def loss(self, batch):

        loss_fn = nn.BCEWithLogitsLoss()
        output = self.forward(batch)
        target = batch['label'].float()
        return loss_fn(output, target)

    def _predict(self, batch):

        output = self.forward(batch)
        scores = torch.sigmoid(output)
        scores = scores.cpu().numpy()[:, 0].tolist()

        return scores

    def predict_fgc(self, batch, threshold=0.5):

        scores = self._predict(batch)
        max_i = 0
        max_score = 0
        sp = []

        for i, score in enumerate(scores):

            if score > max_score:
                max_i = i
                max_score = score
            if score >= threshold:
                sp.append(i)

        if not sp:
            sp.append(max_i)

        return {'sp': sp, 'sp_scores': scores}