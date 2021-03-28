from meantime.loggers import *
# from config import STATE_DICT_KEY, OPTIMIZER_STATE_DICT_KEY, TRAIN_LOADER_RNG_STATE_DICT_KEY
from meantime.config import *
from meantime.utils import AverageMeterSet
from meantime.utils import fix_random_seed_as
from meantime.analyze_table import find_saturation_point

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd

from abc import *
from pathlib import Path
import os
import numpy as np
import time



class AbstractTrainer(metaclass=ABCMeta):
    def __init__(self, args, model, train_loader, val_loader, test_loader, local_export_root):
        print(' ')
        print('meantime / trainers / base.py / AbstractTrainer.__init__')
        
        
        
        self.args = args
        
        # My Code Start
        self.input_middle_seq = args.input_middle_seq
        self.input_middle_num = args.input_middle_num
        self.input_middle_target = args.input_middle_target

        self.input_future_seq = args.input_future_seq
        # My Code End
        
        self.device = args.device
        self.model = model.to(self.device)
        self.use_parallel = args.use_parallel
        if self.use_parallel:
            self.model = nn.DataParallel(self.model)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = self._create_optimizer()
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.decay_step, gamma=args.gamma)
        self.clip_grad_norm = args.clip_grad_norm
        self.epoch_start = 0
        self.best_epoch = self.epoch_start - 1
        self.best_metric_at_best_epoch = -1
        self.accum_iter_start = 0

        self.num_epochs = args.num_epochs
        if self.num_epochs == -1:
            self.num_epochs = 987654321  # Technically Infinite
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric
        self.saturation_wait_epochs = args.saturation_wait_epochs

        self.pilot = args.pilot
        if self.pilot:
            self.num_epochs = 1
            self.pilot_batch_cnt = 1

        self.local_export_root = local_export_root
        self.train_loggers, self.val_loggers, self.test_loggers = self._create_loggers() if not self.pilot else (None, None, None)
        self.add_extra_loggers()

        self.logger_service = LoggerService(args, self.train_loggers, self.val_loggers, self.test_loggers)
        self.log_period_as_iter = args.log_period_as_iter
        
        
        self.resume_training = args.resume_training
        if self.resume_training:
            print('Restoring previous training state')
            self._restore_training_state()
            print('Finished restoring')

    @abstractmethod
    def add_extra_loggers(self):
        pass

    @abstractmethod
    def log_extra_train_info(self, log_data):
        pass

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def calculate_loss(self, batch):
        pass

    @abstractmethod
    def calculate_metrics(self, batch):
        pass

    def train(self):
        print(' ')
        print('meantime / trainers / base.py / AbstractTrainer.train')
        
        


        epoch = self.epoch_start
        best_epoch = self.best_epoch
        accum_iter = self.accum_iter_start
        # self.validate(epoch-1, accum_iter, self.val_loader)
        best_metric = self.best_metric_at_best_epoch
        stop_training = False
        for epoch in range(self.epoch_start, self.num_epochs):
            if self.pilot:
                print('epoch', epoch)
            fix_random_seed_as(epoch)  # fix random seed at every epoch to make it perfectly resumable
            accum_iter = self.train_one_epoch(epoch, accum_iter, self.train_loader)
            print('meantime / trainers / base.py self.train_loader is')
            #print(self.train_loader)
            print('meantime / trainers / base.py train.accum_iter is')
            #print(accum_iter)
            
            self.lr_scheduler.step()  # step before val because state_dict is saved at val. it doesn't affect val result

            val_log_data = self.validate(epoch, accum_iter, mode='val')
            metric = val_log_data[self.best_metric]
            if metric > best_metric:
                best_metric = metric
                best_epoch = epoch
            elif (self.saturation_wait_epochs is not None) and\
                    (epoch - best_epoch >= self.saturation_wait_epochs):
                stop_training = True  # stop training if val perf doesn't improve for saturation_wait_epochs

            if stop_training:
                # load best model
                best_model_logger = self.val_loggers[-1]
                assert isinstance(best_model_logger, BestModelLogger)
                weight_path = best_model_logger.filepath()
                if self.use_parallel:
                    self.model.module.load(weight_path)
                else:
                    self.model.load(weight_path)
                # self.validate(epoch, accum_iter, mode='test')  # test result at best model
                self.validate(best_epoch, accum_iter, mode='test')  # test result at best model
                break

        self.logger_service.complete({
            'state_dict': (self._create_state_dict(epoch, accum_iter)),
        })

    def just_validate(self, mode):
        print(' ')
        print('meantime / trainers / base.py / AbstractTrainer.just_validate')



        ### My Code Start###
        if mode == 'fake_test':
            dummy_epoch, dummy_accum_iter = 0, 0
            self.fake_test(dummy_epoch, dummy_accum_iter, mode)
        elif mode == 'tobigs_test':
            dummy_epoch, dummy_accum_iter = 0, 0
            self.tobigs_test(dummy_epoch, dummy_accum_iter, mode)
        elif mode == 'test_loss':
            dummy_epoch, dummy_accum_iter = 0, 0
            self.test_loss(dummy_epoch, dummy_accum_iter, mode)
        else:
            dummy_epoch, dummy_accum_iter = 0,0
            self.validate(dummy_epoch, dummy_accum_iter, mode)
        ### My Code End

    def train_one_epoch(self, epoch, accum_iter, train_loader, **kwargs):
        print(' ')
        print('meantime / trainers / base.py / AbstractTrainer.train_one_epoch')

        #print(' what is meantime / trainers / base.py / AbstractTrainer, train_loader? is')
        
        #print(train_loader)
        
        
        self.model.train()

        average_meter_set = AverageMeterSet()
        num_instance = 0
        tqdm_dataloader = tqdm(train_loader) if not self.pilot else train_loader

        for batch_idx, batch in enumerate(tqdm_dataloader):
            if self.pilot and batch_idx >= self.pilot_batch_cnt:
                # print('Break training due to pilot mode')
                break
            print(' ')
            print(' ###')
            print(' ')
            print(' raw batch file in training is')
            print(' ')
            #print(batch)
            
            batch_size = next(iter(batch.values())).size(0)
            batch = {k:v.to(self.device) for k, v in batch.items()}
            num_instance += batch_size
            print(' ')
            print(' ')
            print('###############################################')
            print(' meantime / trainers / base.py / AbstractTrainer.train_one_epoch / batch_idx is')
            #print(batch_idx)
            print(' ')
            print(' meantime / trainers / base.py / AbstractTrainer.train_one_epoch / batch is')
            print(' ')
            print('batch')
            print(batch)
            
            print(' ')
            #print("batch['logits'].shape is")
            #print(batch['logits'].shape)
            print(' meantime / trainers / base.py / AbstractTrainer.train_one_epoch / self.optimizer.zero_grad()')
            self.optimizer.zero_grad()
            print(' ')
            print(' meantime / trainers / base.py / AbstractTrainer.train_one_epoch / self.calculate_loss(batch)')
            loss = self.calculate_loss(batch) # 여기로 들어가서 train 과정 자세히 봐야겠다.
            
            
            
            
            
            
            
            
            
            
            
            
            if isinstance(loss, tuple):
                loss, extra_info = loss
                for k, v in extra_info.items():
                    average_meter_set.update(k, v)
            loss.backward()
            
            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            
            self.optimizer.step()
            
            average_meter_set.update('loss', loss.item())
            if not self.pilot:
                tqdm_dataloader.set_description(
                    'Epoch {}, loss {:.3f} '.format(epoch, average_meter_set['loss'].avg))

            accum_iter += batch_size

            if self._needs_to_log(accum_iter):
                if not self.pilot:
                    tqdm_dataloader.set_description('Logging')
                log_data = {
                    # 'state_dict': (self._create_state_dict()),
                    'epoch': epoch,
                    'accum_iter': accum_iter,
                }
                log_data.update(average_meter_set.averages())
                log_data.update(kwargs)
                self.log_extra_train_info(log_data)
                self.logger_service.log_train(log_data)

        log_data = {
            # 'state_dict': (self._create_state_dict()),
            'epoch': epoch,
            'accum_iter': accum_iter,
            'num_train_instance': num_instance,
        }
        log_data.update(average_meter_set.averages())
        log_data.update(kwargs)
        self.log_extra_train_info(log_data)
        self.logger_service.log_train(log_data)
        return accum_iter





    def validate(self, epoch, accum_iter, mode, doLog=True, **kwargs):
        print(' ')
        print('meantime / trainers / base.py / AbstractTrainer.validate is')

        
        
        ### My Code Start###
        my_final_result = -1*torch.ones(1, 205)
        my_dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        my_final_result = my_final_result.to(self.device)
        ### My Code End###
        
        if mode == 'val':
            loader = self.val_loader
        elif mode == 'test':
            loader = self.test_loader
        
        else:
            raise ValueError

        self.model.eval()

        average_meter_set = AverageMeterSet()
        num_instance = 0

        with torch.no_grad():
            tqdm_dataloader = tqdm(loader) if not self.pilot else loader
            for batch_idx, batch in enumerate(tqdm_dataloader):
                if self.pilot and batch_idx >= self.pilot_batch_cnt:
                    # print('Break validation due to pilot mode')
                    break
                batch = {k:v.to(self.device) for k, v in batch.items()}
                batch_size = next(iter(batch.values())).size(0)
                num_instance += batch_size
                
                
                metrics = self.calculate_metrics(batch)
                '''
                print(' ')
                print(' ')
                print('batch idx is')
                print(batch_idx)

                print('batch : token,   [Batch_size x seq_len]')
                print(batch['tokens'])
                print('batch : candidate,   [Batch_size x 100_negative_samples is]')
                print(batch['candidates'])
                print('batch : labels,   [Batch_size x (1 + 100)_labels is]')
                print(batch['labels'])
                ###### MY CODE ######
                #print('epoch is') # 20201214
                #print(epoch)
                #print('batch is') ##### My code 20201119
                #print(batch)
                #print('true answer is')
                #print(batch['candidates'][:,0])
                MY_SCORES, MY_LABELS, MY_CUT, MY_HITS = self.NEW_CODE_PRINT_PREDICTION(batch) ##### My code 20201119
                my_len = len(MY_CUT)
                print("MY_SCORES is,   [Batch_size x (1 + 100)]")
                print(MY_SCORES) ##### My code 20201119
                print(' ')
                #print("MY_LABELS")
                #print(MY_LABELS) ##### My code 20201119
                print("MY_CUT(prediction) is,   [Batch_size x 1]")
                print(MY_CUT) ##### My code 20201119
                print(' ')
                print("MY_HITS is,   [Batch_size x 1]")
                print(MY_HITS) ##### My code 20201119
                print(' ')
                #print('MY_SCORES shape')
                #print(MY_SCORES.shape)
                #print(' ')
                #print('MY_LABELS shape')
                #print(MY_LABELS.shape)
                #print(' ')
                #print('MY_CUT shape')
                #print(MY_CUT.shape)
                #print('MY_HITS.shape')
                #print(MY_HITS.shape)
                '''
                #my_epoch = epoch
                #my_batch_idx = batch_idx
                #my_batch_token = batch['tokens']
                #my_batch_candidate = batch['candidates']
                
                #my_batch_score = MY_SCORES
                #my_batch_cut = MY_CUT
                #my_hit = MY_HITS
                
                #print('true answer is')
                #print(batch['candidates'][:,0])

                #my_epoch1 = torch.Tensor([my_epoch]*batch_size).reshape(batch_size,1)
                #batch_idx1 = torch.Tensor([my_batch_idx]*batch_size).reshape(batch_size,1)
                #batch_idx2 = torch.Tensor(range(batch_size)).reshape(batch_size,1)
                #my_batch_token = my_batch_token.to(self.device)
                #my_candi = batch['candidates'][:,0]
                #my_candi = my_candi.to(self.device)
                #my_cut = MY_CUT
                #my_cut = my_cut.to(self.device)

                #my_epoch1 = my_epoch1.type(my_dtype)
                #batch_idx1 = batch_idx1.type(my_dtype)
                #batch_idx2 = batch_idx2.type(my_dtype)
                #my_batch_token = my_batch_token.type(my_dtype)
                #my_candi = my_candi.type(my_dtype).reshape(batch_size,1)
                #my_hit = my_hit.type(my_dtype)
                #my_cut = my_cut.type(my_dtype)
                
                #print('###')
                #print('my batch token shape')
                #print(my_batch_token.shape)
                #print(my_candi.shape)
                #print(my_hit.shape)
                #print('batch_idx1')
                #print(batch_idx1)
                #print(batch_idx2)
                #print('my_epoch')
                #print(my_epoch)
                
                
                #my_epoch_result = torch.cat([my_epoch1, batch_idx1, batch_idx2, my_batch_token, my_candi, my_cut], 1)
                
                #my_final_result = torch.cat([my_final_result, my_epoch_result], 0)
                ###### MY CODE ######
                
                

                

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                if not self.pilot:
                    description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] +\
                                          ['Recall@%d' % k for k in self.metric_ks[:3]]
                    description = '{}: '.format(mode.capitalize()) + ', '.join(s + ' {:.3f}' for s in description_metrics)
                    description = description.replace('NDCG', 'N').replace('Recall', 'R')
                    description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                    tqdm_dataloader.set_description(description)

            log_data = {
                'state_dict': (self._create_state_dict(epoch, accum_iter)),
                'epoch': epoch,
                'accum_iter': accum_iter,
                'num_eval_instance': num_instance,
            }
            log_data.update(average_meter_set.averages())
            log_data.update(kwargs)
            if doLog:
                if mode == 'val':
                    self.logger_service.log_val(log_data)
                elif mode == 'test':
                    self.logger_service.log_test(log_data)
                else:
                    raise ValueError
                 
                 
        ###### MY CODE ######   
        #ts = time.time()
        #my_final_result = my_final_result.cpu()
        #my_final_result_np = my_final_result.numpy()
        #my_final_result_df = pd.DataFrame(my_final_result_np)
        #FILENAME = 'my_final_result' + mode + str(epoch) + 'time' + str(ts) + '_' +  '.csv'
        #my_final_result_df.to_csv(FILENAME)
        ###### MY CODE ######

        
        return log_data        


    def _create_optimizer(self):
        args = self.args
        if args.optimizer.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            raise ValueError

    def _create_loggers(self):
        train_table_definitions = [
            ('train_log', ['epoch', 'loss'])
        ]
        val_table_definitions = [
            ('val_log', ['epoch'] + \
             ['NDCG@%d' % k for k in self.metric_ks] +
             ['Recall@%d' % k for k in self.metric_ks]),
        ]
        test_table_definitions = [
            ('test_log', ['epoch'] + \
             ['NDCG@%d' % k for k in self.metric_ks] +
             ['Recall@%d' % k for k in self.metric_ks]),
        ]

        train_loggers = [TableLoggersManager(args=self.args, export_root=self.local_export_root, table_definitions=train_table_definitions)]
        val_loggers = [TableLoggersManager(args=self.args, export_root=self.local_export_root, table_definitions=val_table_definitions)]
        test_loggers = [TableLoggersManager(args=self.args, export_root=self.local_export_root, table_definitions=test_table_definitions)]

        if self.local_export_root is not None:
            root = Path(self.local_export_root)
            model_checkpoint = root.joinpath('models')
            val_loggers.append(RecentModelLogger(model_checkpoint))
            val_loggers.append(BestModelLogger(model_checkpoint, metric_key=self.best_metric))

        if USE_WANDB:
            train_loggers.append(WandbLogger(table_definitions=train_table_definitions))
            val_loggers.append(WandbLogger(table_definitions=val_table_definitions, prefix='val_'))
            test_loggers.append(WandbLogger(table_definitions=test_table_definitions, prefix='test_'))

        return train_loggers, val_loggers, test_loggers

    def _create_state_dict(self, epoch, accum_iter):
        return {
            STATE_DICT_KEY: self.model.module.state_dict() if self.use_parallel else self.model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
            SCHEDULER_STATE_DICT_KEY: self.lr_scheduler.state_dict(),
            TRAIN_LOADER_DATASET_RNG_STATE_DICT_KEY: self.train_loader.dataset.get_rng_state(),
            TRAIN_LOADER_SAMPLER_RNG_STATE_DICT_KEY: self.train_loader.sampler.get_rng_state(),
            STEPS_DICT_KEY: (epoch, accum_iter),
        }

    def _restore_best_state(self):
        ### restore best epoch
        df_path = os.path.join(self.local_export_root, 'tables', 'val_log.csv')
        df = pd.read_csv(df_path)
        sat, reached_end = find_saturation_point(df, self.saturation_wait_epochs, display=False)
        e = sat['epoch'].iloc[0]
        self.best_epoch = e
        print('Restored best epoch:', self.best_epoch)

        ###
        state_dict_path = os.path.join(self.local_export_root, 'models', BEST_STATE_DICT_FILENAME)
        chk_dict = torch.load(os.path.abspath(state_dict_path))

        ### sanity check
        _e, _ = chk_dict[STEPS_DICT_KEY]
        assert e == _e

        ### load weights
        d = chk_dict[STATE_DICT_KEY]
        model_state_dict = {}
        # this is for stupid reason
        for k, v in d.items():
            if k.startswith('model.'):
                model_state_dict[k[6:]] = v
            else:
                model_state_dict[k] = v
        if self.use_parallel:
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)

        ### restore best metric
        val_log_data = self.validate(0, 0, mode='val', doLog=False)
        metric = val_log_data[self.best_metric]
        self.best_metric_at_best_epoch = metric
        print('Restored best metric:', self.best_metric_at_best_epoch)

    def _restore_training_state(self):
        self._restore_best_state()

        ###
        state_dict_path = os.path.join(self.local_export_root, 'models', RECENT_STATE_DICT_FILENAME)
        chk_dict = torch.load(os.path.abspath(state_dict_path))

        ### restore epoch, accum_iter
        epoch, accum_iter = chk_dict[STEPS_DICT_KEY]
        self.epoch_start = epoch + 1
        self.accum_iter_start = accum_iter

        ### restore train dataloader rngs
        train_loader_dataset_rng_state = chk_dict[TRAIN_LOADER_DATASET_RNG_STATE_DICT_KEY]
        self.train_loader.dataset.set_rng_state(train_loader_dataset_rng_state)
        train_loader_sampler_rng_state = chk_dict[TRAIN_LOADER_SAMPLER_RNG_STATE_DICT_KEY]
        self.train_loader.sampler.set_rng_state(train_loader_sampler_rng_state)

        ### restore model
        d = chk_dict[STATE_DICT_KEY]
        model_state_dict = {}
        # this is for stupid reason
        for k, v in d.items():
            if k.startswith('model.'):
                model_state_dict[k[6:]] = v
            else:
                model_state_dict[k] = v
        if self.use_parallel:
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)

        ### restore optimizer
        self.optimizer.load_state_dict(chk_dict[OPTIMIZER_STATE_DICT_KEY])

        ### restore lr_scheduler
        self.lr_scheduler.load_state_dict(chk_dict[SCHEDULER_STATE_DICT_KEY])

    def _needs_to_log(self, accum_iter):
        return accum_iter % self.log_period_as_iter < self.args.train_batch_size and accum_iter != 0
        
        
        
    def test_loss(self, epoch, accum_iter, mode, doLog=True, **kwargs):
        
        self.model.eval()
        
        import pickle
        import sys, os 
        import pandas as pd
        
        with open('dataset.pkl', 'rb') as fp:
            data_new = pickle.load(fp)
        item_true_name_dict = data_new['smap']
        item_name_list = [k for k, v in item_true_name_dict.items()]
        
        
        item_type_pickle = pd.read_pickle('type_dict.pkl')
        career_type_list = [k for k, v in item_type_pickle.items() if v in ['중소기업', '스타트업', '대기업']]
        
        # mask value is 1692
        
        # Import Test Data
        import pandas as pd
        test_data = pd.read_csv('final_test_data.csv', index_col = [0])  
        test_data = test_data[['pos1', 'pos2', 'pos3', 'pos4', 'pos5', 'pos6', 'pos7', 'pos8', 'pos9', 'pos10', 'pos11', 'pos12']]
        
        # Make batch, labels arrays
        test_batch = torch.zeros(948, 42)
        test_labels = torch.zeros(948, 42)
        test_batch = test_batch.type(torch.LongTensor)
        test_labels = test_labels.type(torch.LongTensor)
        
        # Make the originally 12th length long data to 40 length -> save it as test_batch(i.e. token)
        #missing_item_count = 0 
        # missing item is 543 or something, out of 948 user x 4~5 item length,
        # so 1 item per each user is unknown item not seen in training process
        
        for row_i in range(1, 949):
            for col_j in range(1, 43):
                
                if col_j < 31:
                    test_data_item = -999
                else:
                    test_data_item = test_data.iloc[row_i - 1, col_j - 30 -1]
                    
                if test_data_item == -999:
                    test_batch[row_i -1, col_j -1] = int(0)
                elif test_data_item in item_name_list:
                    test_batch[row_i -1, col_j - 1] = int(item_true_name_dict[test_data_item])
                else:

                    test_batch[row_i -1, col_j -1] = int(1692)
        
        # Make appropriate Test Labels data, which shows the labels for prediction
        for row_i in range(1, 949):
            for col_j in range(1, 43):
                tmp_index = test_batch[row_i -1, col_j -1]
                if tmp_index in career_type_list:
                    test_labels[row_i - 1, col_j - 1] = int(tmp_index)
                else:
                    test_labels[row_i - 1, col_j - 1] = int(0)
        
        # 
        for row_i in range(1, 949):
            for col_j in range(1, 43):
                tmp_index = test_labels[row_i - 1, col_j - 1]
                if tmp_index != 0:
                    test_batch[row_i - 1, col_j - 1] = 1692
        
        print('test_data is')
        print(' ')
        print(test_data)
        print(' ')
        print('test_data shape is')
        print(' ')
        print(test_data.shape)
        print(' ')
        print('test_batch is')
        print(' ')
        print(test_batch)
        print(' ')
        print('test_batch.shape is')
        print(' ')
        print(test_batch.shape)
        print(' ')
        print('test_labels is')
        print(' ')
        print(test_labels)
        print(' ')
        print('test_labels.shape is')
        print(' ')
        print(test_labels.shape)
        
        with torch.no_grad():
            batch = {'tokens' : test_batch, 'labels' : test_labels}
            print('batch is')
            print(' ')
            print(batch)
            print(' ')
            print('do test_loss = self.calculate_loss(batch)')
            test_valid_index, test_valid_scores, test_valid_labels = self.calculate_loss2(batch)
        
            test_min = test_valid_scores.min()
            test_valid_scores = test_valid_scores + abs(test_min) + 0.01
        
            print('test_valid_scores is')
            print(' ')
            print(test_valid_scores)
            print(' ')
            print('test_valid_scores shape is')
            print(' ')
            print(test_valid_scores.shape)
        
        
            career_type_list = [k for k, v in item_type_pickle.items() if v in ['중소기업', '스타트업', '대기업']]
            
            # Career-item : 1, Non-Career-item : 0
            tmp = torch.zeros(1692, requires_grad=False).cuda()
            for i in range(1, 1692):
                if i in career_type_list:
                    tmp[i-1] = 1
                else:
                    tmp[i-1] = 0
            
            
            tmp = tmp.reshape(1, 1692)
            tmp2 = tmp.repeat(2191, 1)
            #test_valid_scores = torch.mul(test_valid_scores, tmp2)


            #test_valid_labels
            random_negative_mask = tmp2
            
            '''
            # Negative Sample
            for index in range(2191):
                item = test_valid_labels[index]
                for i in range(1, 1693):
                    if i == 1693:
                        random_negative_mask[index, i - 1] = 0

                    if random_negative_mask[index, i -1] == 1:
                        prob = np.random.uniform(0,1,1)
                        if prob < 84./183.:
                            random_negative_mask[index, i - 1] = 0
                if random_negative_mask[index, item - 1] == 0:
                    random_negative_mask[index, item -1 ] = 1
                    
            print('after random masking')
            print(' ')
            print(random_negative_mask.sum(dim = 1))
            '''
            
            #test_valid_scores = torch.mul(test_valid_scores, random_negative_mask)
            test_valid_scores = torch.mul(test_valid_scores, tmp2)


            _, my_indices = torch.max(test_valid_scores, 1)
            my_indices += 1
            
            k = 10
            _, my_indices_topk = torch.topk(test_valid_scores, k = k, dim = 1)
            my_indices_topk += 1
            
            
            print(' ')
            print('score is')
            print((my_indices == test_valid_labels).sum(), 'divided by 2191')
            print(' ')
            top_k_score = torch.zeros(1)
            for i in range(k):
                print('i is ', i)
                increment = (my_indices_topk[:, i] == test_valid_labels).sum()
                print(' ')
                print('increment is')
                print(' ')
                print(increment)
                top_k_score += increment
                
            top_k_score = top_k_score / 2191
            print(' ')
            print('top ', str(k), ' score is')
            print(' ')
            print(top_k_score)
            
            true_items = []
            pred_items = []


            #test_valid_labels = test_valid_labels.reshape(2191, 1)
            #test_valid_scores = test_valid_scores
            
            #test_metrics = self.calculate_metrics2(test_valid_labels, test_valid_scores)

            
            my_indices = my_indices.cpu().detach().numpy()
            test_valid_labels = test_valid_labels.cpu().detach().numpy()
            my_indices_topk = my_indices_topk.cpu().detach().numpy()
            
            
            item_true_name_dict = data_new['smap']
            inv_map = {v: p for p, v in item_true_name_dict.items()}
            
            for true_item in test_valid_labels:
                true_items.append(inv_map[true_item])
            for i in range(k):
                tmp_pred = []
                tmp_row = my_indices_topk[:, i]
                for item in tmp_row:
                    tmp_pred.append(inv_map[item])
                print('i is, ', i)
                print(' ')
                print(tmp_pred)

            print('true item is')
            print(' ')
            print(true_items)
            print(' ')
            print('pred item is')
            print(' ')
            print(pred_items)
            print(' ')
            

            '''
            print('test metrics is')
            print(test_metrics)
            
            print('done till here1')
            for k, v in test_metrics.items():
                average_meter_set.update(k, v)
            print(' ')
            print('done till here2')
            print(' ')
            description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] +\
                                  ['Recall@%d' % k for k in self.metric_ks[:3]]
            description = '{}: '.format(mode.capitalize()) + ', '.join(s + ' {:.3f}' for s in description_metrics)
            description = description.replace('NDCG', 'N').replace('Recall', 'R')
            description = description.format(*(average_meter_set[k].avg for k in description_metrics))            
            
            print('description is')
            print(description)
            print(' ')
            print('the end')
            
            '''

    def tobigs_test(self, epoch, accum_iter, mode, doLog=True, **kwargs):
        if mode == 'val':
            loader = self.val_loader
        elif mode == 'test':
            loader = self.test_loader
        elif mode == 'tobigs_test':
            print('tobigs test')
        
        import pickle
        import sys, os 
        with open('dataset.pkl', 'rb') as fp:
            data_new = pickle.load(fp)
        

        self.model.eval()
        average_meter_set = AverageMeterSet()
        num_instance = 0
        
        input_middle_seq = self.input_middle_seq
        input_middle_num = self.input_middle_num
        input_middle_target = self.input_middle_target
        input_future_seq = self.input_future_seq
        
        if input_middle_seq != None:
            prediction_mode = 'middle'
            input_middle_num = int(input_middle_num[0])
        elif input_future_seq != None:
            prediction_mode = 'future'
        
        
        item_true_name_dict = data_new['smap']
        inv_map = {v: k for k, v in item_true_name_dict.items()}
        
        if prediction_mode == 'middle':
            
            print('##################')
            print(' ')
            print('##################')
            print(' ')
            print('##################')
            print(' ')
            
            print('input_middle_seq is')
            print(' ')
            print(input_middle_seq)
            print(' ')
            '''
            strings = input_middle_seq
            new_strings = []
            for string in strings:
                new_string = string.replace("_", " ")
                new_strings.append(new_string)
            input_middle_seq = new_strings
            '''

            print('input_middle_target is')
            print(' ')
            print(input_middle_target)
            '''
            strings = input_middle_target
            new_strings = []
            for string in strings:
                new_string = string.replace("_", " ")
                new_strings.append(new_string)
            input_middle_target = new_strings
            '''

            
            new_input_seq = [item_true_name_dict[name] for name in input_middle_seq]
            new_input_target = [item_true_name_dict[name] for name in input_middle_target]            

            input_middle_seq = list(map(int, new_input_seq))
            input_middle_target = list(map(int, new_input_target))
            
            pred_seq_len = len(input_middle_seq)
            dummy_tokens = torch.cat([
                torch.tensor( [0]*(42 - (pred_seq_len+2)) ), 
                torch.tensor(input_middle_seq),
                torch.tensor([1692]),
                torch.tensor(input_middle_target)])
            dummy_tokens = torch.cat([dummy_tokens.reshape(1, 42), dummy_tokens.reshape(1, 42)], dim = 0)
            dummy_labels = dummy_tokens*(dummy_tokens == 1692)/1692
            
            batch = {'tokens' : dummy_tokens, 'labels' : dummy_labels}

            with torch.no_grad():
                test_assign = self.calculate_loss2(batch)
                valid_index, valid_scores, valid_labels = test_assign
                
                minimini = valid_scores.min()
                valid_scores = valid_scores + abs(minimini) + 0.01
                scores_all = valid_scores
                
                item_type_pickle = pd.read_pickle('type_dict.pkl')
                nothaksa_list = [k for k, v in item_type_pickle.items() if v not in ['학사', '학사_복전']]
                
                tmp = torch.zeros(1692, requires_grad=False).cuda()
                
                for i in range(1, 1692):
                    if i in nothaksa_list:
                        tmp[i-1] = 1
                    else:
                        tmp[i-1] = 0
                tmp = tmp.reshape(1, 1692)
                tmp2 = torch.cat([tmp, tmp], dim=0)
                
                
                

                valid_scores = torch.mul(valid_scores, tmp2)
                tokens_name1 = []
                for key in dummy_tokens[0]:
                    key = key.cpu().detach().numpy()
                    key = int(key)
                    if key == 0:
                        tokens_name1.append('blank')
                    elif key == 1692:
                        tokens_name1.append('#MASK#')
                    else:
                        tokens_name1.append(inv_map[key])
                tokens_name2 = []
                for key in dummy_tokens[1]:
                    key = key.cpu().detach().numpy()
                    key = int(key)
                    if key == 0:
                        tokens_name2.append('blank')
                    elif key == 1692:
                        tokens_name2.append('#MASK#')
                    else:
                        tokens_name2.append(inv_map[key])

                # Scores all
                _, my_indices = torch.max(valid_scores, 1)
                _, my_indices = torch.topk(valid_scores, k = input_middle_num, dim = 1)
                my_indices = my_indices.cpu().detach().numpy()
                my_indices = my_indices[0]
                
                print('new my indices is')
                print(' ')
                print(my_indices)

                item_true_name_dict = data_new['smap']
                inv_map = {v: k for k, v in item_true_name_dict.items()}
                
                pred_item = []
                pred_item_type = []
                pred_item_name = []
                for key in my_indices:
                    true_key = key + 1
                    pred_item.append(true_key)
                    pred_item_type.append(item_type_pickle[true_key])
                    pred_item_name.append(inv_map[true_key])
                
                print(' ')
                print(' ')
                print('prediction softmax : all item is')
                print(' ')
                print(scores_all)
                print(' ')
                print('shape is')
                print(scores_all.shape)
                print(' ')
                print('pred_item(all) is')
                print(' ')
                print(pred_item)
                print(' ')
                print('pred_item : type is')
                print(' ')
                print(pred_item_type)
                print(' ')
                print('pred_item : name is')
                print(' ')
                print(pred_item_name)
                print(' ')
                print('The End')
                print(' ')
                
                pred_name = pred_item_name
                with open('pred_middle_name.txt', 'w') as f:
                    for name in pred_name:
                        f.write(name+'\n')
                f.close()




            return torch.max(valid_scores, 1)
        
        
        if prediction_mode == 'future':

            print('##################')
            print(' ')
            print('##################')
            print(' ')
            print('##################')
            print(' ')

            
            print('input_future_seq is')
            print(' ')
            '''
            strings = input_future_seq
            new_strings = []
            for string in strings:
                new_string = string.replace("_", " ")
                new_strings.append(new_string)
            input_future_seq = new_strings
            '''
            print(input_future_seq)

            new_input_seq = [item_true_name_dict[name] for name in input_future_seq]
            
            input_future_seq = list(map(int, new_input_seq))

            pred_seq_len = len(input_future_seq)
            dummy_tokens = torch.cat([
                torch.tensor( [0]*(42 - (pred_seq_len+1)) ), 
                torch.tensor(input_future_seq),
                torch.tensor([1692])])
            dummy_tokens = torch.cat([dummy_tokens.reshape(1, 42), dummy_tokens.reshape(1, 42)], dim = 0)
            dummy_labels = dummy_tokens*(dummy_tokens == 1692)/1692
            
            batch = {'tokens' : dummy_tokens, 'labels' : dummy_labels}
            
            with torch.no_grad():
                test_assign = self.calculate_loss2(batch)
                valid_index, valid_scores, valid_labels = test_assign
                
                minimini = valid_scores.min()
                valid_scores = valid_scores + abs(minimini) + 0.01
                scores_all = valid_scores
                
                item_type_pickle = pd.read_pickle('type_dict.pkl')
                career_type_list = [k for k, v in item_type_pickle.items() if v in ['중소기업', '스타트업', '대기업']]
                
                tmp = torch.zeros(1692, requires_grad=False).cuda()
                
                for i in range(1, 1692):
                    if i in career_type_list:
                        tmp[i-1] = 1
                    else:
                        tmp[i-1] = 0
                tmp = tmp.reshape(1, 1692)
                tmp2 = torch.cat([tmp, tmp], dim=0)
                
                valid_scores = torch.mul(valid_scores, tmp2)
                
                tokens_name1 = []
                for key in dummy_tokens[0]:
                    key = key.cpu().detach().numpy()
                    key = int(key)
                    if key == 0:
                        tokens_name1.append('blank')
                    elif key == 1692:
                        tokens_name1.append('#MASK#')
                    else:
                        tokens_name1.append(inv_map[key])
                tokens_name2 = []
                for key in dummy_tokens[1]:
                    key = key.cpu().detach().numpy()
                    key = int(key)
                    if key == 0:
                        tokens_name2.append('blank')
                    elif key == 1692:
                        tokens_name2.append('#MASK#')
                    else:
                        tokens_name2.append(inv_map[key])
                    
                print(tokens_name1)
                print(tokens_name2)
                print(' ')
                print(dummy_tokens)
                print(' ')
                print(dummy_labels)
                

                _, my_indices = torch.max(valid_scores, 1)
                my_indices = my_indices.cpu().detach().numpy()


                pred_item = []
                pred_item_type = []
                pred_item_name = []
                for key in my_indices:
                    true_key = key + 1
                    pred_item.append(true_key)
                    pred_item_type.append(item_type_pickle[true_key])
                    pred_item_name.append(inv_map[true_key])
                print(' ')
                print('pred_item(career) is')
                print(' ')
                print(pred_item[0])
                print(' ')
                print('pred_item : type is')
                print(' ')
                print(pred_item_type[0])
                pred_type = pred_item_type[0]
                print(' ')
                print('pred_item : name is')
                print(' ')
                print(pred_item_name[0])
                
                pred_name = pred_item_name[0]
                with open('pred_future_name.txt', 'w') as f:
                    f.write(pred_name)
                f.close()
                
                print(' ')
                print('The End')
                print(' ')
                
            return torch.max(valid_scores, 1)
            



