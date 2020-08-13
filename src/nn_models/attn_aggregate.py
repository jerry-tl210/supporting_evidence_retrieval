import torch
import torch.nn as nn

class baselineAgg(nn.Module):

    def __init__(self, number_of_sentence, baseline_model):

        super(baselineAgg, self).__init__()
        self.number_of_sentence = number_of_sentence
        self.baseline = baseline_model
        self.softmax = nn.Softmax(dim=1)
        self.linearAgg = nn.Linear(1536, 1)
        self.linear = baseline_model.linear  # nn.Linear(768, 1)

    def forward_nn(self, batch, adjust_weight=False):
        # batch = (batch_size, sentence length)
        # pooler_output = (batch_size, 768)
        # Sent objects into CUDA

        batch_size = batch['input_ids'].shape[0]
        max_sentence_length = batch['input_ids'].shape[2]

        input_ids = batch['input_ids'].view(-1, max_sentence_length)

        device = input_ids.device
        token_type_ids = batch['token_type_ids'].view(-1, max_sentence_length)
        attention_mask = batch['attention_mask'].view(-1, max_sentence_length)
        hidden_state, pooler_output = self.baseline.bert(input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         token_type_ids=token_type_ids)

        # Aggregate
        pooler_output = pooler_output.view(batch_size, -1, 768)  # (batch, 3, 768)

        if adjust_weight:
            target_pair = pooler_output[:, self.number_of_sentence // 2, :].unsqueeze(1)  # (batch, 1, 768)
            target_pair = target_pair.expand(-1, self.number_of_sentence, -1)  # (batch, 3, 768)
            concatenated = torch.cat((target_pair, pooler_output), dim=-1)  # (batch, 3, 768*2)
            weight = self.linearAgg(concatenated)
            weight = self.softmax(weight)
        else:
            weight = torch.tensor([[0.0], [1.0], [0.0]], device=device)

        aggregated_sentence = torch.matmul(weight.transpose(0, 1), pooler_output)  # (batch, 1, 768)
        aggregated_sentence = aggregated_sentence.squeeze(1)  # (batch, 768)

        final_output = self.linear(aggregated_sentence)  # (batch, 1)

        return final_output

    def forward(self, batch):

        output = self.forward_nn(batch)
        labels = batch['label'].type(torch.float)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(output, labels)
        # print("output:", output)
        # print("labels:", labels)
        print("loss:", loss)

        return loss

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