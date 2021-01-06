from transformers import BertPreTrainedModel, BertConfig, BertModel
from transformers.modeling_bert import BertSelfAttention
from torch import nn


class ABert(BertPreTrainedModel):

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attention = BertSelfAttention(config)
        self.cls = nn.Linear(config.hidden_size, 1)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            path_attention_mask=None
    ):
        batch_size, num_padding = input_ids.shape[0], input_ids.shape[1]
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
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
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        output = self.attention(pooled_output.view(batch_size, num_padding, -1), attention_mask=path_attention_mask)
        return self.cls(output[:, 0])
