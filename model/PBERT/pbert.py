from transformers import BertModel, BertPreTrainedModel, BertConfig
from torch import nn
import torch


class PBert(BertPreTrainedModel):
    EPS = 1e-9

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            pool_matrix=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs[0]
        output = torch.bmm(pool_matrix.unsqueeze(1), sequence_output)
        output = self.classifier(output.squeeze())
        if labels is not None:
            return self.loss_fct(output, labels)
        return output

    @classmethod
    def margin_loss_fct(cls, pos_score: torch.Tensor, neg_score: torch.Tensor, margin: torch.Tensor):
        loss = (-(pos_score.squeeze().relu().clamp(min=cls.EPS)) +
                neg_score.squeeze().relu().clamp(min=cls.EPS) +
                margin.squeeze().relu().clamp(min=cls.EPS)).clamp(min=0)
        return loss.sum()
