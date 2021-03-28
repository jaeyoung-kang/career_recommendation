from .bert_base import BertBaseModel
from .embeddings import *
from .bodies import BertBody
from .heads import *

import torch.nn as nn


class BertModel(BertBaseModel):

    def __init__(self, args):
        print(' ')
        print(' meantime / models / transformers_models / bert.py / BertModel.__init__')

        super().__init__(args)
        self.output_info = args.output_info
        self.token_embedding = TokenEmbedding(args)
        self.positional_embedding = PositionalEmbedding(args)
        self.body = BertBody(args)
        if args.headtype == 'dot':
            print(' meantime / models / transformers_models / bert.py / BertModel if args.headtype == "dot" ... ')
            self.head = BertDotProductPredictionHead(args, self.token_embedding.emb)
        elif args.headtype == 'linear':
            print(' meantime / models / transformers_models / bert.py / BertModel if args.headtype == "linear" ...')
            self.head = BertLinearPredictionHead(args)
        else:
            raise ValueError
        self.ln = nn.LayerNorm(args.hidden_units)
        self.dropout = nn.Dropout(p=args.dropout)
        self.init_weights()

    @classmethod
    def code(cls):
        return 'bert'

    def get_logits(self, d):
        
        print(' ')
        print('meantime / models / transformers_models / bert.py / BertModel.get_logits')
        print(' ')
        print('meantime / models / transformers_models / bert.py / BertModel.get_logits / x = d["tokens"] do')
        x = d['tokens']
        print(' ')
        print('meantime / models / transformers_models / bert.py / BertModel.get_logits / x = d["tokens"] is')
        #print(x)
        
        print(' ')
        print('meantime / models / transformers_models / bert.py / BertModel.get_logits / attn_mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1) do')
        attn_mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        print(' ')
        print('meantime / models / transformers_models / bert.py / BertModel.get_logits / attn_mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1) is')
        #print(attn_mask)        
        print(' ')
        print('attn_mask.shape is')
        print(' ')
        #print(attn_mask.shape)
        print(' ')
        print('meantime / models / transformers_models / bert.py / BertModel.get_logits / e = self.token_embedding(d) + self.positional_embedding(d) do')
        print(' ')
        print('token embedding is')
        #print(self.token_embedding(d))
        print(' ')
        print('token embedding shape is')
        print(' ')
        #print(self.token_embedding(d).shape)
        print(' ')
        print('pos embedding is')
        #print(self.positional_embedding(d))
        print(' ')
        #print(self.positional_embedding(d).shape)
        print(' ')
        
        e = self.token_embedding(d) + self.positional_embedding(d)
        e = self.ln(e)
        e = self.dropout(e)  # B x T x H
        print(' ')
        print('finally e is')
        #print(e)
        
        print(' ')
        print('meantime / models / transformers_models / bert.py / BertModel.get_logits / info = {} if self.output_info else None do')
        info = {} if self.output_info else None
        print('meantime / models / transformers_models / bert.py / BertModel.get_logits / info = {} if self.output_info else None is')
        print(' ')        
        #print(info)
        
        print('meantime / models / transformers_models / bert.py / BertModel.get_logits / b = self.body(e, attn_mask, info)  # B x T x H do')
        b = self.body(e, attn_mask, info)  # B x T x H
        print('meantime / models / transformers_models / bert.py / BertModel.get_logits / b = self.body(e, attn_mask, info)  # B x T x H is')
        #print(b)
        print(' ')
        print('b shape is')
        #print(b.shape)
        return b, info

    def get_scores(self, d, logits):
        print(' ')
        print('meantime / models / transformers_models / bert.py / BertModel.get_scores')

        
        
        # logits : B x H or M x H
        if self.training:  # logits : M x H, returns M x V
            h = self.head(logits)  # M x V
        else:  # logits : B x H,  returns B x C
            candidates = d['candidates']  # B x C
            h = self.head(logits, candidates)  # B x C
        return h


    def get_scores2(self, d, logits):
        print(' ')
        print('meantime / models / transformers_models / bert.py / BertModel.get_scores2')

        h = self.head(logits)
        return h
