from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks
from .utils import RETURN_PREDICTION_AND_TRUE_LABELS ##### My code 20201119

class BERTTrainer(AbstractTrainer):
    print(' ')
    print('meantime / trainers / bert.py / BERTTrainer')
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)

    @classmethod
    def code(cls):
        return 'bert'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def calculate_loss(self, batch):
        print(' ')
        print('meantime / trainers / bert.py / BERTTrainer / calculate_loss')

        # loss = self.model(batch, loss=True)
        # loss = loss.mean()
        # return loss
        print('meantime / trainers / bert.py / BERTTrainer / calculate_loss / d = self.model(batch)')
        d = self.model(batch)
        print(' ')
        print('"d" in meantime / trainers / bert.py / BERTTrainer / calculate_loss')
        print(' ')
        #print(d)
        print("d['logits'].shape is")
        print(' ')
        #print(d['logits'].shape)
        print(' ')
        print("d['loss'].shape is")
        print(' ')
        #print(d['loss'].shape)
        print(' ')
        print("d['loss_cnt'].shape is")
        print(' ')
        #print(d['loss_cnt'].shape)
        
        loss, loss_cnt = d['loss'], d['loss_cnt']
        loss = (loss * loss_cnt).sum() / loss_cnt.sum()
        return loss


    def calculate_metrics(self, batch):
        print(' ')
        print('meantime / trainers / bert.py / calculate_metrics')
        print(' ')
        labels = batch['labels']
        scores = self.model(batch)['scores']  # B x C
        # scores = scores.gather(1, candidates)  # B x C

        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics

    def calculate_metrics2(self, labels, scores):
        print(' ')
        print('meantime / trainers / bert.py / calculate_metrics2')
        print(' ')

        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics

    
    def NEW_CODE_PRINT_PREDICTION(self, batch): ##### My code 20201119
        print(' ')
        print('meantime / trainers / bert.py / NEW_CODE_PRINT_PREDICTION')
        labels = batch['labels']
        scores = self.model(batch)['scores']  # B x C
        # scores = scores.gather(1, candidates)  # B x C

        metrics = RETURN_PREDICTION_AND_TRUE_LABELS(scores, labels, self.metric_ks)
        return metrics
        
    def calculate_loss2(self, batch):
        print(' ')
        print('meantime / trainers / bert.py / BERTTrainer / calculate_loss')

        # loss = self.model(batch, loss=True)
        # loss = loss.mean()
        # return loss
        print('meantime / trainers / bert.py / BERTTrainer / calculate_loss / d = self.model(batch)')
        valid_index, valid_scores, valid_labels = self.model(batch)
        
        return valid_index, valid_scores, valid_labels