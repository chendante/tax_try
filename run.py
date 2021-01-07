import argparse
import args_config
import model
import configparser


def _train():
    args_parser = args_config.mine_args_parser()
    args, _ = args_parser.parse_known_args()
    args.taxo_path = "./data/raw_data/TExEval-2_testdata_1.2/gs_taxo/EN/science_wordnet_en.taxo"
    args.epochs = 400
    args.lr = 5e-5
    args.eps = 1e-8
    args.pretrained_path = "bert-base-uncased"
    args.dic_path = "./data/preprocessed/science_dic.json"
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
