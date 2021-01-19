import argparse
import args_config
import model
import configparser
import util


def _train():
    args = dict(taxo_path="./data/raw_data/TExEval-2_testdata_1.2/gs_taxo/EN/environment_eurovoc_en.taxo",
                epochs=50,
                lr=3e-5,
                eps=1e-8,
                dic_path="./data/preprocessed/environment/f_wiki_dic.json",
                padding_max=256,
                pretrained_path="./data/pretrained_model/bert-base-uncased",
                soft_epochs=0,
                margin_beta=0.05,
                r_seed=0,
                epoch_be=20,
                epoch_ave=5)
    args = util.DotDict(args)
    trainer = model.SupervisedTrainer(args)
    trainer.train()


def _eval():
    pass


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('mode', type=str, help="Mode: 'train' or 'eval'")
    args, _ = arg_parser.parse_known_args()

    if args.mode == "train":
        _train()
    elif args.mode == 'eval':
        _eval()
    else:
        raise Exception("Mode not in ['train', 'eval'], e.g. 'python run.py train ...'")
