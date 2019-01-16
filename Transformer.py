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


class Embedding(nn.Module):
    def __init__(self, config, embeddings=None):
        super(Embedding).__init__()

        if embeddings is not None:
            self.word_embeddings = nn.Embedding.from_pretrained(embeddings)
        else:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
