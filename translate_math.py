''' Translate input text with trained model. '''
import os

import torch
import torch.utils.data
import argparse
from tqdm import tqdm
from datetime import datetime

from dataset import collate_fn, TranslationDataset
from transformer.Translator import Translator
from preprocess import read_instances_from_file, convert_instance_to_idx_seq

from corpus_util import Nlp4plpCorpus
from preprocess_math import prepare_instances

from sklearn.metrics import accuracy_score

from main import final_repl


def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True,
                        help='Path to model .pt file')
    parser.add_argument('-data_dir', required=True)
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-dir_out', default="/home/suster/Apps/out/")
    parser.add_argument("--convert-consts", type=str, help="conv | our-map | no-our-map | no. \n/"
                                                               "conv-> txt: -; stats: num_sym+ent_sym.\n/"
                                                               "our-map-> txt: num_sym; stats: num_sym(from map)+ent_sym;\n/"
                                                               "no-our-map-> txt: -; stats: num_sym(from map)+ent_sym;\n/"
                                                               "no-> txt: -; stats: -, only ent_sym;\n/"
                                                               "no-ent-> txt: -; stats: -, no ent_sym;\n/")
    parser.add_argument("--label-type-dec", type=str, default="full-pl",
                            help="predicates | predicates-all | predicates-arguments-all | full-pl | full-pl-no-arg-id | full-pl-split | full-pl-split-plc | full-pl-split-stat-dyn. To use with EncDec.")
    parser.add_argument('-vocab', required=True)
    #parser.add_argument('-output', default='pred.txt',
    #                    help="""Path to output the predictions (each line will
    #                    be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-batch_size', type=int, default=30,
                        help='Batch size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-no_cuda', action='store_true')

    args = parser.parse_args()
    args.cuda = not args.no_cuda

    # Prepare DataLoader
    preprocess_data = torch.load(args.vocab)
    preprocess_settings = preprocess_data['settings']

    if args.convert_consts in {"conv"}:
        assert "nums_mapped" not in args.data_dir
    elif args.convert_consts in {"our-map", "no-our-map", "no", "no-ent"}:
        assert "nums_mapped" in args.data_dir
    else:
        if args.convert_consts is not None:
            raise ValueError
    test_corp = Nlp4plpCorpus(args.data_dir + "test", args.convert_consts)

    if args.debug:
        test_corp.insts = test_corp.insts[:10]
    test_corp.get_labels(label_type=args.label_type_dec)
    test_corp.remove_none_labels()

    # Training set
    test_src_word_insts, test_src_id_insts = prepare_instances(
        test_corp.insts)
    test_tgt_word_insts, test_tgt_id_insts = prepare_instances(
        test_corp.insts, label=True)
    assert test_src_id_insts == test_tgt_id_insts
    test_src_insts = convert_instance_to_idx_seq(test_src_word_insts, preprocess_data['dict']['src'])

    test_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=preprocess_data['dict']['src'],
            tgt_word2idx=preprocess_data['dict']['tgt'],
            src_insts=test_src_insts),
        num_workers=0,
        batch_size=args.batch_size,
        collate_fn=collate_fn)

    translator = Translator(args)

    i = 0
    preds = []
    golds = []

    for batch in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
        all_hyp, all_scores = translator.translate_batch(*batch)
        for idx_seqs in all_hyp:
            for idx_seq in idx_seqs:
                pred = [test_loader.dataset.tgt_idx2word[idx] for idx in idx_seq if test_loader.dataset.tgt_idx2word[idx] != "</s>"]
                gold = [w for w in test_tgt_word_insts[i] if w not in {"<s>", "</s>"}]
                if args.convert_consts == "no":
                    num2n = None
                else:
                    id = test_src_id_insts[i]
                    assert test_corp.insts[i].id == id
                    num2n = test_corp.insts[i].num2n_map
                pred = final_repl(pred, num2n)
                gold = final_repl(gold, num2n)
                preds.append(pred)
                golds.append(gold)
                i += 1
    acc = accuracy_score(golds, preds)
    print(f"Accuracy: {acc:.3f}")
    print("Saving predictions from the best model:")

    assert len(test_src_id_insts) == len(test_src_word_insts) == len(preds) == len(golds)
    f_model = f'{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
    dir_out = f"{args.dir_out}log_w{f_model}/"
    print(f"Save preds dir: {dir_out}")
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    for (id, gold, pred) in zip(test_src_id_insts, golds, preds):
        f_name_t = os.path.basename(f"{id}.pl_t")
        f_name_p = os.path.basename(f"{id}.pl_p")
        with open(dir_out + f_name_t, "w") as f_out_t, open(dir_out + f_name_p, "w") as f_out_p:
            f_out_t.write(gold)
            f_out_p.write(pred)

    #with open(args.output, 'w') as f:
    #   golds
    #    preds
    #    f.write("PRED: " + pred_line + '\n')
    #    f.write("GOLD: " + gold_line + '\n')


    print('[Info] Finished.')


if __name__ == "__main__":
    main()
