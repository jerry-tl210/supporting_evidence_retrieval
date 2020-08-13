from tqdm import tqdm
import logging
from .std import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from . import config

from .preprocess import AttnDataset


from .nn_models.baseline import baseline_model
from .nn_models.attn_aggregate import AttnAggregateModel

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
                             '{train} for training all.')
    parser.add_argument('-num_epochs',
                        default=10,
                        type=int,
                        help='number of epochs')
    parser.add_argument('-batch_size',
                        default=16,
                        type=int,
                        help='batch size')
    parser.add_argument('-model_file_name',
                        required=True,
                        help='output model_file_name')
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
        test_train(args.lr, args.num_epochs, args.batch_size, args.model_file_name)
    elif args.cmd == 'train':
        train(args.lr, args.num_epochs, args.batch_size, args.model_file_name)

def test_use_baseline(train_set, dev_set, model, lr, model_file_name):
    trainer = SER_Trainer(train_set, dev_set, model, lr, model_file_name)
    trainer.train(10, 2)


def test_train(lr, num_epochs, batch_size, model_file_name):
    model = AttnAggregateModel()

    logger.info("Indexing train_set ...")
    train_set = AttnDataset(config.FGC_TRAIN)
    logger.info("train_set has {} instances".format(len(train_set)))

    logger.info("Indexing dev_set")
    dev_set = AttnDataset(config.FGC_DEV)
    logger.info("dev_set has {} instances".format(len(dev_set)))

    trainer = SER_Trainer(train_set[:20], dev_set[:20], model, lr, model_file_name)

    logger.info("Start training ...")
    trainer.train(num_epochs, batch_size)


def train(lr, num_epochs, batch_size, model_file_name):
    model = AttnAggregateModel()

    logger.info("Indexing train_set ...")
    train_set = AttnDataset(config.FGC_TRAIN)
    logger.info("train_set has {} instances".format(len(train_set)))

    logger.info("Indexing dev_set")
    dev_set = AttnDataset(config.FGC_DEV)
    logger.info("dev_set has {} instances".format(len(dev_set)))

    trainer = SER_Trainer(train_set, dev_set, model, lr, model_file_name)

    logger.info("Start training ...")
    trainer.train(num_epochs, batch_size)


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
        self.device = device
        self.n_gpu = n_gpu

        trained_model_path = config.TRAINED_MODELS / model_file_name
        if not os.path.exists(trained_model_path):
            os.mkdir(trained_model_path)
        self.trained_model_path = trained_model_path

    def train(self, num_epochs, batch_size):
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
        num_train_optimization_steps = len(dataloader_train) * num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(
                                                        num_train_optimization_steps * self.warmup_proportion),
                                                    num_training_steps=num_train_optimization_steps)
        for epoch_i in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for batch in tqdm(self.train_set):
                current_loss = self.model(batch)
                current_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                running_loss += current_loss.item()
            learning_rate_scalar = scheduler.get_lr()[0]
            print('lr = %f' % learning_rate_scalar)
            avg_loss = running_loss / len(self.train_set)
            print('epoch %d train_loss: %.3f' % (epoch_i, avg_loss))
            # eval(network, dev_batches, current_epoch, sp_golds, avg_loss)

#     def eval(self, batch_size):
#         self.model.eval()
#         gold_labels = self.dev_set.gold_labels
#         predict_labels = []
#         with torch.no_grad():
#             dataloader_dev = DataLoader(self.dev_set, batch_size=batch_size,
#                                         shuffle=False, collate_fn=hop_baseline_collate_eval,
#                                         num_workers=batch_size)
#             for batch in dataloader_dev:
#                 for key in batch.keys():
#                     batch[key] = batch[key].to(self.device)
#                 predict_next_hop = self.model.module.predict if hasattr(self.model,
#                                                                         'module') else self.model.predict
#                 span_type, hop_start, hop_end, ans_start, ans_end = \
#                     predict_next_hop(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])
#                 for span_type_i, hop_start_i, hop_end_i, ans_start_i, ans_end_i in zip(span_type.split(1, dim=0),
#                                                                                        hop_start.split(1, dim=0),
#                                                                                        hop_end.split(1, dim=0),
#                                                                                        ans_start.split(1, dim=0),
#                                                                                        ans_end.split(1, dim=0)):
#                     if span_type_i.item() == 0:
#                         predict_labels.append(('hop', hop_start_i.item(), hop_end_i.item()))
#                     elif span_type_i.item() == 1:
#                         predict_labels.append(('ans', ans_start_i.item(), ans_end_i.item()))
#                     else:
#                         raise ValueError("predicted span_type error!")
#
#         metrics = eval_span(predict_labels, gold_labels)
#         return metrics
#
# def eval_span(predict_labels, gold_labels):
#     assert len(predict_labels) == len(gold_labels)
#     N = len(predict_labels)
#
#     type_correct = 0
#     span_correct = 0
#     for predict_label, gold_label in zip(predict_labels, gold_labels):
#         if predict_label[0] == gold_label[0]:
#             type_correct += 1
#             if predict_label[1] == gold_label[1].item():
#                 if predict_label[2] == gold_label[2].item():
#                     span_correct += 1
#     return {'span_type_accuracy': type_correct / N, 'span_em': span_correct / N}

if __name__ == '__main__':
    main()