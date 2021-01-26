from transformers.models.bert import BertTokenizer, BertConfig, BertPreTrainedModel, BertModel
from torch import nn
import torch


class DBert(BertPreTrainedModel):
    EPS = 1e-9

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_fct = nn.BCEWithLogitsLoss()
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            pool_matrix=None,
            head_mask=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]
        logits = torch.bmm(pool_matrix.unsqueeze(1), sequence_output)
        if labels is not None:
            return self.loss_fct(logits, labels)
        return logits

    @classmethod
    def margin_loss_fct(cls, pos_score: torch.Tensor, neg_score: torch.Tensor, margin: torch.Tensor):
        loss = (-(pos_score.squeeze().relu()) +
                neg_score.squeeze().relu() +
                margin.squeeze().relu()).clamp(min=0)
        return loss.sum()
