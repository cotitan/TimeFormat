import torch
from torch import nn
import torch.nn.functional as F
import json
import os
from utils import shuffle, BatchManager, build_vocab, load_data


class DotAttention(nn.Module):
    """
    Dot attention calculation
    """
    def __init__(self):
        super(DotAttention, self).__init__()

    def forward(self, enc_states, h_prev):
        """
        calculate the context vector c_t, both the input and output are batch first
        :param enc_states: the encoder states, in shape [batch, seq_len, dim]
        :param h_prev: the previous states of decoder, h_{t-1}, in shape [1, batch, dim]
        :return: c_t: context vector
        """
        alpha_t = torch.bmm(h_prev.transpose(0, 1), enc_states.transpose(1, 2))  # [batch, 1, seq_len]
        alpha_t = F.softmax(alpha_t, dim=-1)
        c_t = torch.bmm(alpha_t, enc_states)  # [batch, 1, dim]
        return c_t


class Model(nn.Module):
    def __init__(self, vocab, out_len=11, emb_dim=32, hid_dim=128):
        super(Model, self).__init__()
        self.out_len = out_len
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.vocab = vocab

        self.embedding_look_up = nn.Embedding(len(self.vocab), self.emb_dim)

        self.encoder = nn.GRU(self.emb_dim, self.hid_dim, batch_first=True)
        self.attention_layer = DotAttention()
        self.decoder = nn.GRU(self.emb_dim + self.hid_dim, self.hid_dim, batch_first=True)

        self.decoder2vocab = nn.Linear(self.hid_dim, len(self.vocab))

        self.loss_layer = nn.CrossEntropyLoss()

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hid_dim)

    def forward(self, inputs, targets, test=False):
        embeds = self.embedding_look_up(inputs)
        states, hidden = self.encoder(embeds, None)
        # logits = self.decode(hidden, targets, test)
        logits = self.attention_decode(states, hidden, targets, test)
        return logits

    def attention_decode(self, enc_states, hidden, targets, test=False):
        if test:
            words = torch.zeros(hidden.shape[1], self.out_len, dtype=torch.long)
            word = torch.ones(hidden.shape[1], dtype=torch.long) * self.vocab["<s>"]
            for i in range(self.out_len):
                embeds = self.embedding_look_up(word).view(-1, 1, self.emb_dim)
                c_t = self.attention_layer(enc_states, hidden)
                outputs, hidden = self.decoder(torch.cat([c_t, embeds], dim=-1), hidden)
                logit = F.tanh(self.decoder2vocab(outputs).squeeze())
                probs = F.softmax(logit, dim=-1)
                word = torch.argmax(probs, dim=-1)
                words[:, i] = word
            return words
        else:
            logits = torch.zeros(hidden.shape[1], self.out_len, len(self.vocab))
            for i in range(self.out_len):
                word = targets[:, i]
                embeds = self.embedding_look_up(word).view(-1, 1, self.emb_dim)
                c_t = self.attention_layer(enc_states, hidden)
                outputs, hidden = self.decoder(torch.cat([c_t, embeds], dim=-1), hidden)
                logits[:, i, :] = self.decoder2vocab(outputs).squeeze()
        return logits


def train(inputs, targets, model, optimizer, batch_size=32, epochs=200):
    inputs_batch_manager = BatchManager(inputs, batch_size)
    targets_batch_manager = BatchManager(targets, batch_size)
    steps = inputs_batch_manager.steps

    for epoch in range(epochs):
        for i in range(steps):
            optimizer.zero_grad()
            batch_inputs = torch.tensor(inputs_batch_manager.next_batch(), dtype=torch.long)
            batch_targets = torch.tensor(targets_batch_manager.next_batch(), dtype=torch.long)
            logits = model(batch_inputs, batch_targets)  # exclude start token
            loss = model.loss_layer(logits.transpose(1, 2), batch_targets[:, 1:])
            loss.backward()
            optimizer.step()
        print(loss)

    torch.save(model.state_dict(), os.path.join("models", "params.pkl"))


def test(inputs, targets, vocab, model, dataset="dev"):
    id2w = {v: k for (k, v) in vocab.items()}
    inputs1 = torch.tensor(inputs, dtype=torch.long)
    targets1 = torch.tensor(targets, dtype=torch.long)
    words = model(inputs1, targets1, test=True)
    right_count = 0
    badcases = []
    for i in range(len(targets)):
        pred = words[i].numpy()

        src_sent = "".join([id2w[id] for id in inputs[i] if id2w[id] != "<pad>"])
        pred_sent = "".join([id2w[id] for id in pred[:-1]])
        target_sent = "".join([id2w[id] for id in targets[i][:-1] if id != 0])
        if pred_sent == target_sent:
            right_count += 1
        else:
            badcases.append([src_sent, pred_sent, target_sent])
        if dataset == "dev":
            print(src_sent, pred_sent, target_sent, sep="\t\t")
    print("accuracy on %s data = %f" % (dataset, right_count / len(targets)))
    return badcases

# TODO
# data augmentation on 2014-2017


if __name__ == "__main__":
    vocab, max_src_len, max_tgt_len, inputs, targets = load_data(n_data=850)
    inputs, targets = shuffle(inputs, targets)

    idx = int(len(inputs) * 0.8)
    model = Model(vocab)
    # if os.path.exists("models/params_190.pkl"):
    #     model.load_state_dict(torch.load("models/params_190.pkl"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    train(inputs[:idx], targets[:idx], model, optimizer, epochs=50)
    badcases_dev = test(inputs[idx:], targets[idx:], vocab, model, dataset="dev")
    badcases_train = test(inputs[:idx], targets[:idx], vocab, model, dataset="train")
    print("************** Badcases on training data ********************")
    for c in badcases_train:
        print(c[0], c[1], c[2], sep="\t\t")
    print("************** Badcases on dev data ********************")
    for c in badcases_dev:
        print(c[0], c[1], c[2], sep="\t\t")

