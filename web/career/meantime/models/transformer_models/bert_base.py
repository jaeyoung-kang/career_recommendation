from meantime.models.base import BaseModel

import torch.nn as nn

from abc import *

#My Code Start
#from meantime.trainers.base import MY_MODE
#My Code End

class BertBaseModel(BaseModel, metaclass=ABCMeta):
    print(' ')
    print('meantime / models / transformers_model / bert_base.py / BertBaseModel')

    def __init__(self, args):
        print(' ')
        print(' meantime / models / transformers_model / bert_base.py / BertBaseModel.__init__')
        
        
        print('BertBaseModel self.args is')
        print(args)
        
        global MY_MODE
        MY_MODE = args.mode
        
        super().__init__(args)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, d):
        print(' ')
        print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.forward')
        
        print(' ')
        
        print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.forward / logits, info = self.get_logits(d)')
        print(' ')

        logits, info = self.get_logits(d)
        print(' ')
        print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.forward / logits, info = self.get_logits(d) (ret) is')
        ret = {'logits':logits, 'info':info}
        #print(ret)
        
        print(' ')
        print(' ret["logits"].shape is')
        #print(' ')
        #print(ret['logits'].shape)


        if self.training:
            print(' ')
            print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.forward / if self.training...')
            print(' ')
            print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.forward / if self.training / d')
            print(' ')
            #print(d)
            print(' ')
            print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.forward / if self.training / labels =  d"labels"]')            
            labels = d['labels']
            #print(labels)
            
            
            loss, loss_cnt = self.get_loss(d, logits, labels)
            ret['loss'] = loss
            ret['loss_cnt'] = loss_cnt
            print(' ')
            print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.forward / if self.training / ret["loss"] = loss, ret["loss_cnt"] = loss_cnt')
            return ret
        
        
        elif MY_MODE == 'tobigs_test':
            print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.forward / elif self.my_test_middle')
            labels = d['labels']
            valid_index, valid_scores, valid_labels = self.get_loss2(d, logits, labels)
            return valid_index, valid_scores, valid_labels 

        elif MY_MODE == 'test_loss':
            print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.forward / elif self.test_loss')
            labels = d['labels']
            valid_index, valid_scores, valid_labels = self.get_loss2(d, logits, labels)
            return valid_index, valid_scores, valid_labels 

            
        else:
            print(' ')
            print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.forward / if self.training else...')

            # get scores (B x V) for validation
            last_logits = logits[:, -1, :]  # B x H
            ret['scores'] = self.get_scores(d, last_logits)  # B x C
            return ret
            
            

        


    @abstractmethod
    def get_logits(self, d):
        pass

    @abstractmethod
    def get_scores(self, d, logits):  # logits : B x H or M x H, returns B x C or M x V
        pass

    @abstractmethod
    def get_scores2(self, d, logits):  # logits : B x H or M x H, returns B x C or M x V
        pass



    def get_loss(self, d, logits, labels):
        print(' ')
        print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.get_loss')

        _logits = logits.view(-1, logits.size(-1))  # BT x H
        _labels = labels.view(-1)  # BT

        print(' ')
        print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.get_loss / _logits = logits.view(-1, logits.size(-1))')
        print(' ')
        #print(_logits)
        print(' ')
        print('_logits.shape is')
        #print(_logits.shape)
        print(' ')
        print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.get_loss / _labels = labels.view(-1)')
        print(' ')
        #print(_labels)
        print(' ')
        print('_labels.shape is')
        #print(_labels.shape)
        print(' ')

        print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.get_loss / valid = _labels > 0')
        valid = _labels > 0
        print(' ')
        #print(valid)
        print(' ')
        print('valid shape is')
        print(' ')
        #print(valid.shape)
        print(' ')
        loss_cnt = valid.sum()  # = M
        print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.get_loss / loss_cnt = valid.sum()  # = M')
        print(' ')
        #print(loss_cnt)
        print(' ')
        #print(loss_cnt.shape)
        print(' ')
        valid_index = valid.nonzero().squeeze()  # M
        print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.get_loss / valid_index = valid.nonzero().squeeze()  # M')
        print(' ')
        #print(valid_index)
        print(' ')
        print('valid_index shape is')
        #print(valid_index.shape)
        print(' ')
        print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.get_loss / valid_logits = _logits[valid_index]  # M x H')
        print(' ')
        valid_logits = _logits[valid_index]  # M x H
        #print(valid_logits)
        print(' ')
        print('valid_logits shape is')
        print(' ')
        #print(valid_logits.shape)
        print(' ')
        print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.get_loss / valid_scores = self.get_scores(d, valid_logits)  # M x V')
        valid_scores = self.get_scores(d, valid_logits)  # M x V
        
        
        
        print(' ')
        #print(valid_scores)
        print(' ')
        print('valid_scores shape is')
        print(' ')
        #print(valid_scores.shape)
        print(' ')
        print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.get_loss / valid_labels = _labels[valid_index]  # M')        
        print(' ')
        valid_labels = _labels[valid_index]  # M
        
        #print(valid_labels)
        print(' ')
        print('valid_labels shape is')
        print(' ')
        #print(valid_labels.shape)
        
        print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.get_loss / loss = self.ce(valid_scores, valid_labels)')        
        print(' ')
        loss = self.ce(valid_scores, valid_labels)
        
        #print(loss)
        print(' ')
        print('loss shape is')
        print(' ')
        #print(loss.shape)
        print(' ')
        print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.get_loss / loss, loss_cnt = loss.unsqueeze(0), loss_cnt.unsqueeze(0)')       
        print(' ')
        loss, loss_cnt = loss.unsqueeze(0), loss_cnt.unsqueeze(0)
        
        print('loss is')
        print(' ')
        #print(loss)
        print(' ')
        print('loss shape is')
        #print(loss.shape)

        print('loss_cnt is')
        print(' ')
        #print(loss_cnt)
        print(' ')
        print('loss_cnt shape is')
        #print(loss_cnt.shape)
        
        
        
        return loss, loss_cnt


# My Code Start
    def get_loss2(self, d, logits, labels):
        print(' ')
        print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.get_loss')

        _logits = logits.view(-1, logits.size(-1))  # BT x H
        _labels = labels.view(-1)  # BT

        print(' ')
        print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.get_loss / _logits = logits.view(-1, logits.size(-1))')
        print(' ')
        print(_logits)
        print(' ')
        print('_logits.shape is')
        print(_logits.shape)
        print(' ')
        print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.get_loss / _labels = labels.view(-1)')
        print(' ')
        print(_labels)
        print(' ')
        print('_labels.shape is')
        print(_labels.shape)
        print(' ')

        print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.get_loss / valid = _labels > 0')
        valid = _labels > 0
        print(' ')
        print(valid)
        print(' ')
        print('valid shape is')
        print(' ')
        print(valid.shape)
        print(' ')
        loss_cnt = valid.sum()  # = M
        print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.get_loss / loss_cnt = valid.sum()  # = M')
        print(' ')
        print(loss_cnt)
        print(' ')
        print(loss_cnt.shape)
        print(' ')
        valid_index = valid.nonzero().squeeze()  # M
        print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.get_loss / valid_index = valid.nonzero().squeeze()  # M')
        print(' ')
        print(valid_index)
        print(' ')
        print('valid_index shape is')
        print(valid_index.shape)
        print(' ')
        print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.get_loss / valid_logits = _logits[valid_index]  # M x H')
        print(' ')
        valid_logits = _logits[valid_index]  # M x H
        print(valid_logits)
        print(' ')
        print('valid_logits shape is')
        print(' ')
        print(valid_logits.shape)
        print(' ')
        print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.get_loss / valid_scores = self.get_scores(d, valid_logits)  # M x V')
        valid_scores = self.get_scores2(d, valid_logits)  # M x V
        
        
        
        print(' ')
        print(valid_scores)
        print(' ')
        print('valid_scores shape is')
        print(' ')
        print(valid_scores.shape)
        print(' ')
        print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.get_loss / valid_labels = _labels[valid_index]  # M')        
        print(' ')
        valid_labels = _labels[valid_index]  # M
        
        print(valid_labels)
        print(' ')
        print('valid_labels shape is')
        print(' ')
        print(valid_labels.shape)
        
        print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.get_loss / loss = self.ce(valid_scores, valid_labels)')        
        print(' ')
        loss = self.ce(valid_scores, valid_labels)
        
        print(loss)
        print(' ')
        print('loss shape is')
        print(' ')
        print(loss.shape)
        print(' ')
        print(' meantime / models / transformers_models / bert_base.py / BertBaseModel.get_loss / loss, loss_cnt = loss.unsqueeze(0), loss_cnt.unsqueeze(0)')       
        print(' ')
        loss, loss_cnt = loss.unsqueeze(0), loss_cnt.unsqueeze(0)
        
        print('loss is')
        print(' ')
        print(loss)
        print(' ')
        print('loss shape is')
        print(loss.shape)

        print('loss_cnt is')
        print(' ')
        print(loss_cnt)
        print(' ')
        print('loss_cnt shape is')
        print(loss_cnt.shape)
        
        
        
        return valid_index, valid_scores, valid_labels
# My Code End


