import json
import os
import random

class BatchManager:
    def __init__(self, datas, batch_size):
        self.steps = int(len(datas) / batch_size)
        if self.steps * batch_size < len(datas):
            self.steps += 1
        self.datas = datas
        self.bs = batch_size
        self.bid = 0

    def next_batch(self):
        batch = list(self.datas[self.bid * self.bs: (self.bid + 1) * self.bs])
        self.bid += 1
        if self.bid == self.steps:
            self.bid = 0
        return batch


def build_vocab(vocab_file="vocab.json"):
    vocab = {"<s>": 0, "</s>": 1, "<pad>": 2}
    fin = open("date_lines.txt", "r", encoding="utf8")
    for line in fin:
        for ch in line:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    json.dump(vocab, open(vocab_file, "w", encoding="utf8"))



def load_data(vocab_file="vocab.json", n_data=850, include_dirty_data=False):
    if not os.path.exists(vocab_file):
        build_vocab(vocab_file)
    vocab = json.load(open(vocab_file, "r", encoding="utf8"))
    fin = open("date_lines.txt", "r", encoding="utf8")
    n = 0
    inputs = []
    targets = []
    max_src_len = 0
    for idx, line in enumerate(fin):
        try:
            src, target = line.strip().split("|")
        except Exception:
            print(idx, line)
            continue
        if len(target) != 10:
            print(idx, target)
        if len(src) > max_src_len:
            max_src_len = len(src)
        if target != "xxxx-xx-xx":
            inputs.append(src)
            targets.append(target)
        n += 1
        if n == n_data:
            break
    final_inputs = []
    for line in inputs:
        line = [vocab['<s>']] + [vocab[ch] for ch in line] + [vocab['</s>']]
        line += [vocab["<pad>"]] * (max_src_len + 2 - len(line))
        final_inputs.append(line)
    targets = [[vocab[ch] for ch in line] for line in targets]
    targets = [[vocab["<s>"]] + line + [vocab["</s>"]] for line in targets ]
    max_tgt_len = len(targets[-1])
    return vocab, max_src_len, max_tgt_len, final_inputs, targets


def shuffle(inputs, targets):
    c = list(zip(inputs, targets))
    random.shuffle(c)
    inputs, targets = zip(*c)
    return list(inputs), list(targets)

