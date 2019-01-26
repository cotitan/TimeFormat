import torch
from Transformer import Transformer, Encoder, Decoder
from utils import load_data, build_vocab, BatchManager, shuffle
from torch.optim import Adam, lr_scheduler
# from tensorboardX import SummaryWriter
import os


config = {
    "n_src_layers": 6,
    "n_tgt_layers": 6,
    "n_head": 8,
    "d_word_emb": 32,
    "d_k": 32,
    "d_model": 128
}


def train(inputs, targets, model, optimizer, n_epochs, scheduler=None):
    batch_x = BatchManager(inputs, 32)
    batch_y = BatchManager(targets, 32)
    
    # writer = SummaryWriter()
    k = 0
    for epoch in range(n_epochs):
        for i in range(batch_x.steps):
            optimizer.zero_grad()

            x = torch.tensor(batch_x.next_batch(), dtype=torch.long).cuda()
            y = torch.tensor(batch_y.next_batch(), dtype=torch.long).cuda()
            
            logits = model(x, y)

            loss = 0
            for j in range(y.shape[0]):
                loss += model.loss_layer(logits[j], y[j,1:])
            loss /= y.shape[0]
            
            if i % 1 == 0:
                _loss_ = float(loss.cpu().detach().numpy())
                # writer.add_scalar('scalar/train_loss', _loss_, k)
                k += 1
                print('epoch %d, step %d, loss = %f' % (epoch, i, _loss_))

            loss.backward()
            optimizer.step()
        # if scheduler is not None:
        #     scheduler.step()

        torch.save(model.state_dict(), 'models/params_transformer.pkl')
    # writer.close()


def main():
    vocab, max_src_len, max_tgt_len, inputs, targets = load_data('vocab.json', n_data=850)
    inputs, targets = shuffle(inputs, targets)
    
    # set d_model = d_word_vec
    model = Transformer(n_src_vocab=len(vocab), n_tgt_vocab=len(vocab),
                max_src_len=max_src_len, max_tgt_len=max_tgt_len, d_word_vec=32,
                N=6, n_head=4, d_q=8, d_k=8, d_v=8, d_model=32, d_inner=64)
    model.cuda()
    # model = Encoder(len(vocab), max_src_len, d_src_emb=32, N=3, n_head=4,
    #                 d_q=16, d_k=16, d_v=16, d_model=32, d_inner=32)

    model_file = 'models/params_transformer.pkl'
    if os.path.exists(model_file):
        print("Loading parameters from %s" % model_file)
        model.load_state_dict(torch.load(model_file))

    parameters = filter(lambda p : p.requires_grad, model.parameters())
    optimizer = Adam(parameters, lr=0.00001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

    idx = int(len(inputs) * 0.9)
    train(inputs[:idx], targets[:idx], model, optimizer, n_epochs=20, scheduler=scheduler)
    eval(model, vocab, inputs[idx:], targets[idx:], out_len=12)


def visualize(x_ids, y_ids, pred_ids, vocab):
    x_ids = x_ids.cpu().numpy()
    y_ids = y_ids.cpu().numpy()
    pred_ids = pred_ids.cpu().numpy()

    id2w = {k:v for v, k in vocab.items()}
    for i in range(x_ids.shape[0]):
        x = [id2w[idx] for idx in x_ids[i][1:] if idx not in [vocab['<pad>'], vocab['</s>']]]
        x = "".join(x)
        y = [id2w[idx] for idx in y_ids[i][1:-1]] # exclude <s> and </s>
        y = "".join(y)
        pred = [id2w[idx] for idx in pred_ids[i][1:-1]]
        pred = "".join(pred)

        print(x, y, pred)


def eval(model, vocab, inputs, targets, out_len=12):
    # TODO: 解决预测问题
    model.eval()
    batch_x = BatchManager(inputs, 16)
    batch_y = BatchManager(targets, 16)
    for i in range(batch_x.steps):
        x = torch.tensor(batch_x.next_batch(), dtype=torch.long).cuda()
        y = torch.tensor(batch_y.next_batch(), dtype=torch.long).cuda()
        
        tgt_seq = torch.ones(x.shape[0], out_len, dtype=torch.long).cuda()
        tgt_seq *= vocab['<pad>']
        tgt_seq[:, 0] = vocab['<s>']
        tgt_seq[:, -1] = vocab['</s>']
        for j in range(1, out_len):
            logits = model(x, tgt_seq)
            last_word = torch.argmax(logits[:,j-1,:], dim=-1).view(-1, 1)
            tgt_seq[:,j] = last_word.squeeze()
        visualize(x, y, tgt_seq, vocab)

    model.train()

if __name__ == '__main__':
    main()
