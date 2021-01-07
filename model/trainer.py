import torch
import model
from torch.utils.data import dataloader
import transformers
from tqdm import tqdm


class SupervisedTrainer(object):
    def __init__(self, args):
        # super(BaseTrainer, self).__init__(args)
        self.args = args
        self.model = model.DBert.from_pretrained(args.pretrained_path,
                                                 gradient_checkpointing=True,
                                                 output_attentions=False,  # 模型是否返回 attentions weights.
                                                 output_hidden_states=False,  # 模型是否返回所有隐层状态.
                                                 )
        self.input_reader = model.InputReader(args.taxo_path)
        self.sampler = model.Sampler(self.input_reader.taxo_pairs,
                                     tokenizer=transformers.BertTokenizer.from_pretrained(args.pretrained_path),
                                     dic_path=args.dic_path,
                                     padding_max=64)

    def train(self):
        optimizer = transformers.AdamW(self.model.parameters(),
                                       lr=self.args.lr,  # args.learning_rate - default is 5e-5
                                       eps=self.args.eps  # args.adam_epsilon  - default is 1e-8
                                       )
        data_loader = dataloader.DataLoader(self.sampler, batch_size=16, shuffle=True, drop_last=True)
        # 创建学习率调度器
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=0,
                                                                 num_training_steps=len(data_loader) * self.args.epochs)
        self.model.cuda()
        for epoch in range(self.args.epochs):
            self.model.train()
            self.sampler.sample_paths()  # 重新sample negative path
            loss_all = 0.0
            for batch in tqdm(data_loader, desc='Train epoch %s' % epoch, total=len(data_loader)):
                optimizer.zero_grad()
                pos_output = self.model(input_ids=batch["pos_ids"].cuda(), token_type_ids=batch["pos_type_ids"].cuda(),
                                        attention_mask=batch["pos_attn_masks"].cuda())
                neg_output = self.model(input_ids=batch["neg_ids"].cuda(), token_type_ids=batch["neg_type_ids"].cuda(),
                                        attention_mask=batch["neg_attn_masks"].cuda())
                loss = self.model.margin_loss_fct(pos_output, neg_output, batch["margin"].cuda())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                optimizer.step()
                scheduler.step()
                loss_all += loss.item()
            print(epoch, loss_all)
            if epoch < 2:
                continue
            self.model.eval()
            testing_data = self.sampler.get_eval_data()
            count = 0
            mrr = 0
            for node, data in testing_data.items():
                with torch.no_grad():
                    output = self.model(input_ids=data["ids"].cuda(), token_type_ids=data["token_type_ids"].cuda(),
                                        attention_mask=data["attn_masks"].cuda())
                index = output.squeeze().argmax().cpu()
                _, indices = output.squeeze().sort(descending=True)
                rank = (indices == data["label"]).nonzero().squeeze()
                if index == data["label"]:
                    count += 1
                mrr += 1.0 / (float(rank) + 1.0)
            print("acc:", count / float(len(testing_data)))
            print("mrr:", mrr / float(len(testing_data)))
