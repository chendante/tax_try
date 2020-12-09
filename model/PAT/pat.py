from torch import nn
import bpemb
import torch


class Pat(nn.Module):
    pad = "<pad>"
    EPS = 1e-9

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.tokenizer = bpemb.BPEmb(lang='en', dim=embed_dim, vs=10000, add_pad_emb=True)
        self.embedding = nn.Embedding.from_pretrained(embeddings=torch.FloatTensor(self.tokenizer.emb.vectors),
                                                      freeze=False)
        self.att_layer = nn.MultiheadAttention(embed_dim, num_heads=num_heads)
        self.W = nn.Linear(embed_dim, 1)

    def forward(self, ids, pool_matrix: torch.Tensor, padding_masks):
        """
        :param pool_matrix: shape(batch_size, padding)
        :param padding_masks: shape(batch_size, padding)
        :param ids: shape(batch_size, padding)
        :return:
        """
        output = self.embedding(ids)
        output = output.transpose(0, 1)
        # output: (padding, batch_size, embedding)
        output = self.att_layer(output, output, output, key_padding_mask=padding_masks, need_weights=False)[0]
        output = torch.bmm(pool_matrix.unsqueeze(1), output.transpose(0, 1))
        output = self.W(output.squeeze())
        return output

    @classmethod
    def loss_fuc(cls, pos_score: torch.Tensor, neg_score: torch.Tensor, margin: torch.Tensor):
        loss = (-(pos_score.squeeze().relu().clamp(min=cls.EPS)) +
                neg_score.squeeze().relu().clamp(min=cls.EPS) +
                margin.squeeze().relu().clamp(min=cls.EPS)).clamp(min=0)
        return loss.sum()

    @classmethod
    def loss_fuc_no_margin(cls, pos_score: torch.Tensor, neg_score: torch.Tensor):
        return (-(pos_score.squeeze().relu().clamp(min=cls.EPS)) +
                neg_score.squeeze().relu().clamp(min=cls.EPS)).clamp(min=0).sum()


if __name__ == '__main__':
    b = bpemb.BPEmb(lang='en', add_pad_emb=True)
    print(b.emb.vocab.get("<pad>").index)
    print(b.encode_ids(["Stratford", "anarchism", "<pad>"]))
