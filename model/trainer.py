import torch
from model.base.trainer import BaseTrainer
import model
from tqdm import tqdm
from model import util
import transformers

_MODELS = {
    'crim': model.Crim,
}


def get_inner_model_init_func(model_type, k):
    return lambda x: _MODELS[model_type](x, k)


class SupervisedTrainer(BaseTrainer):
    def __init__(self, args):
        super(BaseTrainer, self).__init__(args)
        self.model: model.Frame = model.Frame.from_pretrained_embedding(args.embedding_file_path,
                                                                 get_inner_model_init_func(args.model_type,
                                                                                           args.crim_k))
        self.input_reader = model.InputReader(args.data_path, args.gold_path)
        self.sampler = model.Sampler(self.input_reader.d_words, self.input_reader.g_words, self.model,
                                     batch_size=args.batch_size, padding_max=args.padding_max)

    def train(self):
        emb_optimizer = torch.optim.SparseAdam(self.model.embedding.parameters(), lr=self.args.emb_lr)
        gen_optimizer = torch.optim.Adam(self.model.inner_module.parameters(), lr=self.args.gen_lr)
        optimizer = util.CombinedOptimizer(emb_optimizer, gen_optimizer)
        self.model.cuda()
        for epoch in range(self.args.epochs):
            for batch in tqdm(self.sampler, desc='Train epoch %s' % epoch, total=len(self.sampler)):
                optimizer.zero_grad()
                query = batch[0].cuda()
                hypers = batch[1].cuda()
                attention_masks = batch[2].cuda()
                targets = batch[3].cuda()
                output = self.model(query, hypers)
                loss = self.model.loss_fct(output, targets, attention_masks)
                loss.backward()
                optimizer.step()

    def save_model(self):



    def _training(self):
        pass
