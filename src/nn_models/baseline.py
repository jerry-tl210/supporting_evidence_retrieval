import torch
import torch.nn as nn
from transformers import BertModel


class BaselineModel(nn.Module):

    def __init__(self):

        super(BaselineModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.linear = nn.Linear(768, 1)

    def forward_nn(self, batch):
        # batch['ids'] = (batch_size, 1, sent_len)
        # batch['segment_ids'] = (batch_size, 1, sent_len)
        # batch['mask_ids'] = (batch_size, 1, sent_len)
        # pooler_output = (batch_size, 768)
        # output = (batch_size, 1)

        input_ids = batch['input_ids']# (batch_size, sent_len)
        attention_mask = batch['attention_mask']# (batch_size, sent_len)
        token_type_ids = batch['token_type_ids'] # (batch_size, sent_len)
        
        hidden_state, pooler_output = self.bert(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                token_type_ids=token_type_ids)
        
        linear_output = self.linear(pooler_output)
        return linear_output

    def forward(self, batch):

        output = self.forward_nn(batch)
        labels = batch['label'].type(torch.float)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(output, labels)
        return loss

    def predict(self, batch, threshold=0.5):
        output = self.forward_nn(batch)
        score = torch.sigmoid(output).cpu()
        predict_label = torch.where(score > threshold, torch.ones(len(score),1), torch.zeros(len(score), 1))
        predict_label = predict_label.numpy().astype(int).tolist()
        return predict_label, score.numpy().tolist()
