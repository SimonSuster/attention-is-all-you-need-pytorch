''' Handling the data io '''
import argparse
import torch
import transformer.Constants as Constants
from corpus_util import Nlp4plpCorpus


def prepare_instances(insts, max_sent_len=1e9, label=False):
    ''' Convert instances into word seq lists and vocab '''

    word_insts = []
    id_insts = []
    removed_sent_count = 0
    for inst in insts:
        if label:
            words = inst.label
        else:
            words = inst.txt
        if len(words) > max_sent_len:
            removed_sent_count += 1
            word_insts += [None]
            id_insts.append(None)
            continue
        word_insts += [[Constants.BOS_WORD] + words + [Constants.EOS_WORD]]
        id_insts.append(inst.id)

    print('[Info] Get {} instances'.format(len(word_insts)))

    if removed_sent_count > 0:
        print('[Warning] {} instances are removed as they exceeded the max sentence length {}.'
              .format(removed_sent_count, max_sent_len))

    assert len(word_insts) == len(id_insts), print(f"{len(word_insts)}, {len(id_insts)}")
    return word_insts, id_insts

def build_vocab_idx(word_insts, min_word_count):
    ''' Trim vocab by number of occurence '''

    full_vocab = set(w for sent in word_insts for w in sent)
    print('[Info] Original Vocabulary size =', len(full_vocab))

    word2idx = {
        Constants.BOS_WORD: Constants.BOS,
        Constants.EOS_WORD: Constants.EOS,
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK}

    word_count = {w: 0 for w in full_vocab}

    for sent in word_insts:
        for word in sent:
            word_count[word] += 1

    ignored_word_count = 0
    for word, count in word_count.items():
        if word not in word2idx:
            if count > min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1

    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx

def convert_instance_to_idx_seq(word_insts, word2idx):
    ''' Mapping words to idx sequence. '''
    return [[word2idx.get(w, Constants.UNK) for w in s] for s in word_insts]

def main():
    ''' Main function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', required=True)
    parser.add_argument('-save_data', required=True)
    parser.add_argument("--convert-consts", type=str, help="conv | our-map | no-our-map | no. \n/"
                                                               "conv-> txt: -; stats: num_sym+ent_sym.\n/"
                                                               "our-map-> txt: num_sym; stats: num_sym(from map)+ent_sym;\n/"
                                                               "no-our-map-> txt: -; stats: num_sym(from map)+ent_sym;\n/"
                                                               "no-> txt: -; stats: -, only ent_sym;\n/"
                                                               "no-ent-> txt: -; stats: -, no ent_sym;\n/")
    parser.add_argument("--label-type-dec", type=str, default="full-pl",
                            help="predicates | predicates-all | predicates-arguments-all | full-pl | full-pl-no-arg-id | full-pl-split | full-pl-split-plc | full-pl-split-stat-dyn. To use with EncDec.")
    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=200)
    parser.add_argument('-min_word_count', type=int, default=2)
    parser.add_argument('-share_vocab', action='store_true')
    parser.add_argument('-vocab', default=None)

    args = parser.parse_args()

    if args.convert_consts in {"conv"}:
        assert "nums_mapped" not in args.data_dir
    elif args.convert_consts in {"our-map", "no-our-map", "no", "no-ent"}:
        assert "nums_mapped" in args.data_dir
    else:
        if args.convert_consts is not None:
            raise ValueError
    train_corp = Nlp4plpCorpus(args.data_dir + "train", args.convert_consts)
    print(f"Size of train: {len(train_corp.insts)}")
    dev_corp = Nlp4plpCorpus(args.data_dir + "dev", args.convert_consts)
    test_corp = Nlp4plpCorpus(args.data_dir + "test", args.convert_consts)

    debug = False
    if debug:
        train_corp.fs = train_corp.fs[:10]
        train_corp.insts = train_corp.insts[:10]
        dev_corp.fs = dev_corp.fs[:10]
        dev_corp.insts = dev_corp.insts[:10]
        test_corp.insts = test_corp.insts[:10]

    train_corp.get_labels(label_type=args.label_type_dec, max_output_len=1000)
    dev_corp.get_labels(label_type=args.label_type_dec, max_output_len=1000)
    test_corp.get_labels(label_type=args.label_type_dec)

    train_corp.remove_none_labels()
    dev_corp.remove_none_labels()
    test_corp.remove_none_labels()

    args.max_token_seq_len = args.max_word_seq_len + 2 # include the <s> and </s>

    # Training set
    train_src_word_insts, _ = prepare_instances(
        train_corp.insts, args.max_word_seq_len)
    train_tgt_word_insts, _ = prepare_instances(
        train_corp.insts, args.max_word_seq_len, label=True)

    if len(train_src_word_insts) != len(train_tgt_word_insts):
        print('[Warning] The training instance count is not equal.')
        min_inst_count = min(len(train_src_word_insts), len(train_tgt_word_insts))
        train_src_word_insts = train_src_word_insts[:min_inst_count]
        train_tgt_word_insts = train_tgt_word_insts[:min_inst_count]

    #- Remove empty instances
    train_src_word_insts, train_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(train_src_word_insts, train_tgt_word_insts) if s and t]))

    # Validation set
    valid_src_word_insts, _ = prepare_instances(
        dev_corp.insts, args.max_word_seq_len)
    valid_tgt_word_insts, _ = prepare_instances(
        dev_corp.insts, args.max_word_seq_len, label=True)

    if len(valid_src_word_insts) != len(valid_tgt_word_insts):
        print('[Warning] The validation instance count is not equal.')
        min_inst_count = min(len(valid_src_word_insts), len(valid_tgt_word_insts))
        valid_src_word_insts = valid_src_word_insts[:min_inst_count]
        valid_tgt_word_insts = valid_tgt_word_insts[:min_inst_count]

    #- Remove empty instances
    valid_src_word_insts, valid_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(valid_src_word_insts, valid_tgt_word_insts) if s and t]))

    # Build vocabulary
    if args.vocab:
        predefined_data = torch.load(args.vocab)
        assert 'dict' in predefined_data

        print('[Info] Pre-defined vocabulary found.')
        src_word2idx = predefined_data['dict']['src']
        tgt_word2idx = predefined_data['dict']['tgt']
    else:
        if args.share_vocab:
            print('[Info] Build shared vocabulary for source and target.')
            word2idx = build_vocab_idx(
                train_src_word_insts + train_tgt_word_insts, args.min_word_count)
            src_word2idx = tgt_word2idx = word2idx
        else:
            print('[Info] Build vocabulary for source.')
            src_word2idx = build_vocab_idx(train_src_word_insts, args.min_word_count)
            print('[Info] Build vocabulary for target.')
            tgt_word2idx = build_vocab_idx(train_tgt_word_insts, args.min_word_count)

    # word to index
    print('[Info] Convert source word instances into sequences of word index.')
    train_src_insts = convert_instance_to_idx_seq(train_src_word_insts, src_word2idx)
    valid_src_insts = convert_instance_to_idx_seq(valid_src_word_insts, src_word2idx)

    print('[Info] Convert target word instances into sequences of word index.')
    train_tgt_insts = convert_instance_to_idx_seq(train_tgt_word_insts, tgt_word2idx)
    valid_tgt_insts = convert_instance_to_idx_seq(valid_tgt_word_insts, tgt_word2idx)

    data = {
        'settings': args,
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx},
        'train': {
            'src': train_src_insts,
            'tgt': train_tgt_insts},
        'valid': {
            'src': valid_src_insts,
            'tgt': valid_tgt_insts}}

    print('[Info] Dumping the processed data to pickle file', args.save_data)
    torch.save(data, args.save_data)
    print('[Info] Finish.')

if __name__ == '__main__':
    main()
