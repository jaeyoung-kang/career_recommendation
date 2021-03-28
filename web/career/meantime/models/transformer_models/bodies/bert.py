from torch import nn as nn

from .transformers.transformer import TransformerBlock


class BertBody(nn.Module):
    def __init__(self, args):
        print(' ')
        print('MEANTIME / meantime / transformers_model / bodies / bert.py / BertBody.__init__')
        super().__init__()

        n_layers = args.num_blocks

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(args) for _ in range(n_layers)])

    def forward(self, x, attn_mask, info=None):
        print(' ')
        print('MEANTIME / meantime / transformers_model / bodies / bert.py / BertBody.forward(self, x, attn_mask, info)')
        print(' ')
        print('x is')
        #print(x)
        print(' ')
        print('x shape is')
        print(' ')
        #print(x.shape)
        print(' ')
        print('attn_mask is')
        print(' ')
        #print(attn_mask)
        print(' ')
        print('attn_mask shape is')
        print(' ')
        #print(attn_mask.shape)
        print(' ')
        print('info is')
        #print(info)
        print(' ')
        print('info shape is')
        #print(info.shape)
        print(' ')        
        print('MEANTIME / meantime / transformers_model / bodies / bert.py / BertBody.forward(self, x, attn_mask, info) / for loop do ')
        loopcount = 0
        for layer, transformer in enumerate(self.transformer_blocks):
            loopcount += 1
            print(' ')
            print('looping count is')
           # print(loopcount)
            x = transformer.forward(x, attn_mask, layer, info)
            print(' ')
            print('x after loop is')
            print(' ')
            #print(x)
            print('x shape is')
            #print(x.shape)
        print(' ')
        print('MEANTIME / meantime / transformers_model / bodies / bert.py / BertBody.forward(self, x, attn_mask, info) / for loop / x is ')
        print(' ')
        print('finally x and shape is')
        #print(x)
        print(' ')
        #print(x.shape)
        return x
