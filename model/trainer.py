import torch
from model.base.trainer import BaseTrainer
import model
from tqdm import tqdm
from model import util
from model.PAT.pat import Pat
from torch.utils.data import dataloader


class SupervisedTrainer(object):
    def __init__(self, args):
        # super(BaseTrainer, self).__init__(args)
        self.args = args
        self.model: Pat = Pat(args.embed_dim, args.num_heads)
        self.input_reader = model.InputReader(args.taxo_path)
        self.sampler = model.Sampler(self.input_reader.taxo_pairs, self.model.tokenizer, padding_max=64)

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10,
        #                                                        verbose=True)
        self.model.cuda()
        for epoch in range(self.args.epochs):
            self.model.train()
            self.sampler.sample_paths()  # 重新sample negative path
            data_loader = dataloader.DataLoader(self.sampler, batch_size=32, shuffle=True, drop_last=True)
            # for batch in tqdm(data_loader, desc='Train epoch %s' % epoch, total=len(self.sampler)/32):
            loss_all = 0.0
            for batch in data_loader:
                optimizer.zero_grad()
                pos_output = self.model(batch["pos_ids"].cuda(), batch["pos_pool_matrix"].cuda(),
                                        batch["pos_attn_masks"].cuda())
                neg_output = self.model(batch["neg_ids"].cuda(), batch["neg_pool_matrix"].cuda(),
                                        batch["neg_attn_masks"].cuda())
                # if epoch < self.args.epochs / 10:
                #     loss = self.model.loss_fuc_no_margin(pos_output, neg_output)
                # else:
                loss = self.model.loss_fuc(pos_output, neg_output, batch["margin"].cuda())
                # pos_output = self.model(batch["pos_ids"], batch["pos_pool_matrix"],
                #                         batch["pos_attn_masks"])
                # neg_output = self.model(batch["neg_ids"], batch["neg_pool_matrix"],
                #                         batch["neg_attn_masks"])
                # loss = self.model.loss_fuc(pos_output, neg_output, batch["margin"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                optimizer.step()
                loss_all += loss.item()
                # print(loss.item())
            print(epoch, loss_all)
            if epoch < 20 or epoch % 5 != 0:
                continue
            self.model.eval()
            testing_data = self.sampler.get_eval_data()
            count = 0
            mrr = 0
            for node, data in testing_data.items():
                with torch.no_grad():
                    output = self.model(data["ids"].cuda(), data["pool_matrix"].cuda(), data["attn_masks"].cuda())
                index = output.squeeze().argmax().cpu()
                _, indices = output.squeeze().sort(descending=True)
                rank = (indices == data["label"]).nonzero().squeeze()
                if index == data["label"]:
                    count += 1
                mrr += 1.0 / (float(rank) + 1.0)
            print("acc:", count / float(len(testing_data)))
            print("mrr:", mrr / float(len(testing_data)))
