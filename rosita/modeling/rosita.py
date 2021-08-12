import math, torch
import torch.nn as nn

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class FeedForward(nn.Module):
    def __init__(self, cfg, model_dict):
        super(FeedForward, self).__init__()
        self.dense0 = nn.Linear(model_dict['HSIZE'], model_dict['HFF'])
        self.dense1 = nn.Linear(model_dict['HFF'], model_dict['HSIZE'])

    def forward(self, x):
        return self.dense1(gelu(self.dense0(x)))


class MHAtt(nn.Module):
    def __init__(self, cfg, model_dict):
        super(MHAtt, self).__init__()
        self.cfg = cfg
        self.model_dict = model_dict

        self.dense_v = nn.Linear(model_dict['HSIZE'], model_dict['HSIZE'])
        self.dense_k = nn.Linear(model_dict['HSIZE'], model_dict['HSIZE'])
        self.dense_q = nn.Linear(model_dict['HSIZE'], model_dict['HSIZE'])
        self.dense_merge = nn.Linear(model_dict['HSIZE'], model_dict['HSIZE'])
        self.dropout = nn.Dropout(cfg.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)
        v = self.dense_v(v).view(n_batches, -1, self.model_dict['HHEAD'], self.model_dict['HBASE']).transpose(1, 2)
        k = self.dense_k(k).view(n_batches, -1, self.model_dict['HHEAD'], self.model_dict['HBASE']).transpose(1, 2)
        q = self.dense_q(q).view(n_batches, -1, self.model_dict['HHEAD'], self.model_dict['HBASE']).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(n_batches, -1, self.model_dict['HSIZE'])
        atted = self.dense_merge(atted)
        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # bert style masking
        scores = scores + mask
        att_map = torch.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)
        return torch.matmul(att_map, value)


class SelfAtt(nn.Module):
    def __init__(self, cfg, model_dict):
        super(SelfAtt, self).__init__()
        self.mhatt = MHAtt(cfg, model_dict)
        self.ffn = FeedForward(cfg, model_dict)

        self.dropout0 = nn.Dropout(cfg.DROPOUT_R)
        self.layer_norm0 = LayerNorm(model_dict['HSIZE'], eps=1e-12)
        self.dropout1 = nn.Dropout(cfg.DROPOUT_R)
        self.layer_norm1 = LayerNorm(model_dict['HSIZE'], eps=1e-12)

    def forward(self, x, x_mask):
        x = self.layer_norm0(x + self.dropout0(self.mhatt(x, x, x, x_mask)))
        x = self.layer_norm1(x + self.dropout1(self.ffn(x)))
        return x


class Backbone(nn.Module):
    def __init__(self, cfg, model_dict):
        super(Backbone, self).__init__()
        self.layers = nn.ModuleList([SelfAtt(cfg, model_dict) for _ in range(model_dict['LAYER'])])

    def forward(self, x, x_mask):
        for layer in self.layers:
            x = layer(x, x_mask)
        return x


class TextEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, cfg, vocab_size):
        super(TextEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, cfg.HSIZE)
        self.position_embeddings = nn.Embedding(cfg.POS_EMB_IN_SIZE, cfg.HSIZE)
        self.token_type_embeddings = nn.Embedding(cfg.TYPE_EMB_IN_SIZE, cfg.HSIZE)

        self.layer_norm = LayerNorm(cfg.HSIZE, eps=1e-12)
        self.dropout = nn.Dropout(cfg.DROPOUT_R)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class VisualEmbeddings(nn.Module):
    def __init__(self, cfg):
        super(VisualEmbeddings, self).__init__()
        self.cfg = cfg
        self.dense_imgfeat = nn.Linear(cfg.IMGFEAT_SIZE, cfg.HSIZE)
        if cfg.USE_BBOXFEAT:
            self.dense_bboxfeat = nn.Linear(cfg.BBOXFEAT_SIZE, cfg.HSIZE)
        self.layer_norm = LayerNorm(cfg.HSIZE, eps=1e-12)
        self.dropout = nn.Dropout(cfg.DROPOUT_R)

    def forward(self, imgfeat, bboxfeat):
        imgfeat = self.dense_imgfeat(imgfeat)
        if self.cfg.USE_BBOXFEAT:
            bboxfeat = self.dense_bboxfeat(bboxfeat)
            imgfeat += bboxfeat
        imgfeat = self.layer_norm(imgfeat)
        imgfeat = self.dropout(imgfeat)

        return imgfeat


class Pooler(nn.Module):
    def __init__(self, cfg):
        super(Pooler, self).__init__()
        self.dense = nn.Linear(cfg.HSIZE, cfg.HSIZE)

    def forward(self, x):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = x[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = torch.tanh(pooled_output)
        return pooled_output


class MmITMHead(nn.Module):
    def __init__(self, cfg):
        super(MmITMHead, self).__init__()
        self.cfg = cfg
        self.dense = nn.Linear(cfg.HSIZE, 2)
    
    def forward(self, x_pooled):
        pred = self.dense(x_pooled)
        return pred
