import torch
from torch import nn
import json


class ScaledDotAttention(nn.Module):
    def __init__(self):
        super(ScaledDotAttention, self).__init__()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, q, k, v):
        """
        perform scaled-dot-attention, given query,key,value
        :params q: query, size=[batch, 1, d_q]
        :params k: key, size=[batch, seqlen, d_q]
        :params v: value, size=[batch, seqlen, d_v]
        :return attn_vec:
        :return attn_weight:
        """
        scale = 1 / torch.sqrt(q.size(-1))
        # [batch, 1, d_q] * [batch, d_q, seqlen] *  ==> [batch, 1, seqlen]
        attn_weight = self.softmax(torch.bmm(q, k.transpose(1,2)) * scale)
        # [batch, 1, seqlen] * [batch, seqlen, d_v] ==> [batch, 1, d_v]
        attn_vec = torch.bmm(attn_weight, v)
        return attn_vec, attn_weight


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(hidden_size))
        self.shift = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
    
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = (x - mu).powe(2).mean(-1, keep_dim=True)
        x = (x - mu) / torch.sqrt(sigma + self.eps) # avoid sqrt(0)
        return self.scale * x + self.shift


class Embedding(nn.Module):
    def __init__(self, config, embeddings=None):
        super(Embedding, self).__init__()

        if embeddings is not None:
            self.word_embeddings = nn.Embedding.from_pretrained(embeddings)
        else:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, input_ids):
        pass  # TODO


class MultiHeadAttention(nn.Module):
    def __init__(self, num_head=8, d_model=512, d_q=64, d_k=64, d_v=64):
        super(MultiHeadAttention, self).__init__()
        assert d_q == d_k, "dim(key) must be equal to dim(query)"

        self.num_head = num_head

        self.W_Q = [nn.Linear(d_q, d_model) for _ in range(num_head)]
        self.W_K = [nn.Linear(d_k, d_model) for _ in range(num_head)]
        self.W_V = [nn.Linear(d_v, d_model) for _ in range(num_head)]

        self.attn_layer = ScaledDotAttention()
        self.linear_out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v):
        Qs = [self.W_Q(q) for _ in range(self.num_head)]
        Ks = [self.W_K(k) for _ in range(self.num_head)]
        Vs = [self.W_V(v) for _ in range(self.num_head)]
        heads = [self.attn_layer(Qs[i], Ks[i], Vs[i]) for i in range(self.num_head)]
        out = self.linear_out(torch.cat(heads))
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # TODO
        pass
    
    def forward(nn.Module):
        # TODO
        pass


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # TODO
        pass
    
    def forward(self):
        # TODO


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        # TODO
    
    def forward(self):
        # TODO

