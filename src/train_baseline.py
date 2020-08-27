from tqdm import tqdm
import logging
from .std import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from . import config
import glob

from .preprocess import AttnDataset


from .nn_models.baseline import BaselineModel
from .nn_models.attn_aggregate import AttnAggregateModel
from .nn_models.multiBERTs import MultiBERTsModel

logger = logging.getLogger(__name__)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-llv',
                        default='INFO',
                        help='Logging level')
    parser.add_argument('-log',
                        default=None,
                        help='Output log file')
    parser.add_argument('-lr',
                        default=5e-5,
                        type=float,
                        help='Learning rate')
    parser.add_argument('-cmd',
                        required=True,
                        help='Command: {test_train} for testing.\n'
                             '{train} for training all.'
                             '{eval_all} for evaluating all')
    parser.add_argument('-num_epochs',
                        default=10,
                        type=int,
                        help='number of epochs')
    parser.add_argument('-batch_size',
                        default=16,
                        type=int,
                        help='batch size')
    parser.add_argument('-accumulation_steps',
                        type=int)
    parser.add_argument('-model_file_name',
                        required=True,
                        help='output model_file_name')
    parser.add_argument('-data_type',
                        required=True)   
    parser.add_argument('-adjust_weight',
                        default=False,
                        action='store_true')
    parser.add_argument('-transform',
                        default=False,
                        action='store_true')
    parser.add_argument('-multiBERTs',
                        default=False,
                        action='store_true')
    parser.add_argument('acc_gradient',
                        default=False,
                        action='store_true')
    
    args = parser.parse_args()

    myLogFormat = '%(asctime)s **%(levelname)s** [%(name)s:%(lineno)s] - %(message)s'
    logging.basicConfig(level=str2llv(args.llv), format=myLogFormat, datefmt='%Y/%m/%d %H:%M:%S')
    if args.log:
        myhandlers = log_w(args.log)
        logger.addHandler(myhandlers)
        logger.log(100, ' '.join(sys.argv))
    else:
        logger.log(100, ' '.join(sys.argv))

    if args.cmd == 'test_train':
        test_train(args.lr, args.num_epochs, args.batch_size, 
                   args.model_file_name, args.data_type, args.adjust_weight,
                   args.transform, args.accumulation_steps, args.multiBERTs,
                   args.acc_gradient)
    elif args.cmd == 'train':
        train(args.lr, args.num_epochs, args.batch_size, 
              args.model_file_name, args.data_type, args.adjust_weight,
              args.transform, args.accumulation_steps, args.multiBERTs,
              args.acc_gradient


def test_train(lr, num_epochs, batch_size, model_file_name, data_type, adjust_weight, transform, accumulation_steps, multiBERTs, acc_gradient):
    
    baseline = BaselineModel()
    
    logger.info("Indexing train_set ...")
    
    if data_type == 'fgc':
        train_data = pd.read_json(config.FGC_TRAIN)
        train_instances, train_ids, train_segment_ids, train_mask_ids, train_labels = AttnDataset(train_data[:10], data_type, multiBERTs)
    if data_type == 'ssqa':
        train_data = []
        for filename in glob.glob('*.json'):
            with open(filename) as json_file:
                train_data = train_data + json.load(json_file)
        train_set = AttnDataset(train_data[:10], data_type, multiBERTs)
    logger.info("train_set has {} instances".format(len(train_set)))

    logger.info("Indexing dev_set")
    if data_type == 'fgc': 
        dev_data = json_load(config.FGC_DEV)
        dev_set = AttnDataset(dev_data[:10], data_type, multiBERTs)
    if data_type == 'ssqa':
        dev_data = []
        # Remember to edit the path!
        for filename in glob.glob('*.json'):
            with open(filename) as json_file:
                dev_data = dev_data + json.load(json_file)
        dev_set = AttnDataset(dev_data[:1], data_type, multiBERTs)   
    logger.info("dev_set has {} instances".format(len(dev_set)))
    
    trainer = SER_Trainer(train_set, dev_set, model, lr, model_file_name)

    logger.info("Start training ...")
    trainer.train(num_epochs, batch_size, accumulation_steps, acc_gradient)


def train(lr, num_epochs, batch_size, model_file_name, data_type, adjust_weight, transform, accumulation_steps, multiBERTs, acc_gradient):
    baseline = BaselineModel()
    
    logger.info("Indexing train_set ...")
    if data_type == 'fgc':
        train_data = json_load(config.FGC_TRAIN)
        train_set = AttnDataset(train_data, data_type, multiBERTs)
    if data_type == 'ssqa':
        train_data = []
        for filename in glob.glob('*.json'):
            with open(filename) as json_file:
                train_data = train_data + json.load(json_file)
        train_set = AttnDataset(train_data, data_type, multiBERTs)
        
    logger.info("train_set has {} instances".format(len(train_set)))

    logger.info("Indexing dev_set")
    if data_type == 'fgc':   
        dev_data = json_load(config.FGC_DEV)
        dev_set = AttnDataset(dev_data, data_type, multiBERTs)
    if data_type == 'ssqa':
        dev_data = []
        # Remember to edit the path!
        for filename in glob.glob('*.json'):
            with open(filename) as json_file:
                dev_data = dev_data + json.load(json_file)
        dev_set = AttnDataset(dev_data, data_type, multiBERTs)  
    logger.info("dev_set has {} instances".format(len(dev_set)))

    trainer = SER_Trainer(train_set, dev_set, model, lr, model_file_name)

    logger.info("Start training ...")
    trainer.train(num_epochs, batch_size, accumulation_steps, acc_gradient)


class SER_Trainer:
    def __init__(self, train_set, dev_set, model, lr, model_file_name):
        self.train_set = train_set
        self.dev_set = dev_set
        self.model = model
        self.lr = lr
        self.eval_freq = 1
        self.warmup_proportion = 0.1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        logger.info("SER_Trainer in device: {}".format(device))
        self.device = device
        self.n_gpu = n_gpu

        trained_model_path = config.TRAINED_MODELS / model_file_name
        if not os.path.exists(trained_model_path):
            os.mkdir(trained_model_path)
        self.trained_model_path = trained_model_path

    def train(self, num_epochs, batch_size, accumulation_steps=1, acc_gradient, show_batch=True):
        
        logger.info("batch_size:{} accumulate_steps:{} acc_gradient:{}".format(batch_size, accumulation_steps, acc_gradient))
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        dataloader_train = DataLoader(self.train_set, batch_size=batch_size, shuffle=True, num_workers=batch_size)
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        # model
        self.model.to(self.device)

        if self.n_gpu > 1:
            self.model = nn.DataParallel(self.model)

        # optimizer
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        num_train_optimization_steps = len(dataloader_train) * num_epochs / accumulation_steps
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(
                                                        num_train_optimization_steps * self.warmup_proportion),
                                                    num_training_steps=num_train_optimization_steps)
        logger.info("start training loop")
        

        if acc_gradient:   
            batch_eval = 100
            for epoch_i in range(num_epochs):
                self.model.train()
                total_loss = 0
                logger.info("train epoch_i:{}".format(epoch_i))

                if acc_gradient:
                for batch_i, batch in enumerate(tqdm(dataloader_train)):
                    #logger.info('batch_i:{}'.format(batch_i))
                    for t_i, t in batch.items():
                        batch[t_i] = t.to(self.device)

                    loss = self.model(batch)
                    if self.n_gpu > 1:
                        loss = loss.mean()
                    loss = loss / accumulation_steps
                    loss.backward()
                    total_loss += loss.item()

                    if (batch_i+1) % accumulation_steps == 0:

                        torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                    if (batch_i % batch_eval == 0):
                        print("current batch loss:", loss.item())
                        print("total loss:", total_loss)

                learning_rate_scalar = scheduler.get_lr()[0]
                print('lr = %f' % learning_rate_scalar)
                avg_loss = total_loss / len(dataloader_train)
                print('epoch %d train_loss: %.3f' % (epoch_i, avg_loss))
                print("---------------------dev set performance----------------------")
                self.eval(epoch_i, batch_size, self.dev_set, avg_loss)
                print("---------------------train set performance----------------------")
                self.eval(epoch_i, batch_size, self.train_set, avg_loss)
        
        else:
            
            self.model.train()
            running_loss = 0.0
            for batch in tqdm(dataloader_train):
                for key in batch.keys():
                    batch[key] = batch[key].to(self.device)
                
                current_loss = self.model(batch, adjust_weight=True)
                if self.n_gpu > 1:
                    current_loss = current_loss.mean()
                current_loss.sum().backward()
                torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                running_loss += current_loss.item()
            learning_rate_scalar = scheduler.get_lr()[0]
            print('lr = %f' % learning_rate_scalar)
            avg_loss = running_loss / len(dataloader_train)
            print('epoch %d train_loss: %.3f' % (epoch_i, avg_loss))
            print("---------------------dev set performance----------------------")
            self.eval(epoch_i, batch_size, self.dev_set, avg_loss)
            print("---------------------train set performance----------------------")
            self.eval(epoch_i, batch_size, self.train_set, avg_loss)
            
    def eval(self, epoch_i, batch_size, dataset, avg_loss):
        self.model.eval()
        cumulative_len = dataset.cumulative_len
        indices_golds = dataset.shints
        with torch.no_grad():

            counter = 0
            dataloader = DataLoader(dataset, batch_size=batch_size,
                                        shuffle=False, num_workers=batch_size)
            indices_preds = []
            current_document_labels = []
            for batch in tqdm(dataloader):
                for key in batch.keys():
                    batch[key] = batch[key].to(self.device)
                
                predict_se = self.model.module.predict if hasattr(self.model,
                                                                  'module') else self.model.predict
                current_labels = predict_se(batch)
                for label in current_labels:
                    if counter + 1 in cumulative_len:
                        current_document_labels += label
                        current_document_indices = np.where(np.array(current_document_labels) == 1)[0].tolist()
                        indices_preds.append(current_document_indices)
                        current_document_labels = []
                        
                    else:
                        current_document_labels += label

                    counter = counter + 1

            print("indices_golds", len(indices_golds))
            print("indices_preds", len(indices_preds))
        metrics = self.eval_sp(indices_golds, indices_preds)
        print(indices_golds)
        print(indices_preds)
        #torch.save(self.model.state_dict(), "Models_SEs/attn_aggregate/model_epoch{0}_eval_em:{1:.3f}_precision:{2:.3f}_recall:{3:.3f}_f1:{4:.3f}_loss:{5:.3f}.m".format(epoch_i, metrics['sp_em'], metrics['sp_prec'], metrics['sp_recall'], metrics['sp_f1'], avg_loss))

        return metrics

    def update_sp(self, metrics, sp_gold, sp_pred):

        tp, fp, fn = 0, 0, 0

        for p in sp_pred:
            if p in sp_gold:
                tp += 1
            else:
                fp += 1
        for g in sp_gold:
            if g not in sp_pred:
                fn += 1

        precision = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        em = 1.0 if fp + fn == 0 else 0.0

        metrics['sp_em'] += em
        metrics['sp_f1'] += f1
        metrics['sp_prec'] += precision
        metrics['sp_recall'] += recall

        return precision, recall, f1

    def eval_sp(self, indices_golds, indices_preds):

        metrics = {'sp_em': 0, 'sp_prec': 0, 'sp_recall': 0, 'sp_f1': 0}

        assert len(indices_golds) == len(indices_preds)

        for sp_gold, sp_pred in zip(indices_golds, indices_preds):
            self.update_sp(metrics, sp_gold, sp_pred)

        N = len(indices_golds)
        for k in metrics.keys():
            metrics[k] /= N
            metrics[k] = round(metrics[k], 3)
        print(metrics)
        return metrics


if __name__ == '__main__':
    main()