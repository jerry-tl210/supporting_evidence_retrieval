import torch
from .nn_models.attn_aggregate import AttnAggregateModel
from .nn_models.baseline import BaselineModel
from .evaluation.eval_metric import eval_sp
from .evaluation.analysis import get_analysis
from .std import *
from . import config

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
    parser.add_argument('-model_file_name',
                        required=True,
                        help='trained model_file_name')
    parser.add_argument('-data_type',
                        required=True)
    parser.add_argument('-adjust_weight',
                        default=False,
                        action='store_true')
    parser.add_argument('-transform',
                        default=False,
                        action='store_true')
    parser.add_argument('-batch_size',
                        default=16,
                        type=int,
                        help='batch size')
    args = parser.parse_args()

    myLogFormat = '%(asctime)s **%(levelname)s** [%(name)s:%(lineno)s] - %(message)s'
    logging.basicConfig(level=str2llv(args.llv), format=myLogFormat, datefmt='%Y/%m/%d %H:%M:%S')
    if args.log:
        myhandlers = log_w(args.log)
        logger.addHandler(myhandlers)
        logger.log(100, ' '.join(sys.argv))
    else:
        logger.log(100, ' '.join(sys.argv))
    
    trained_baseline_model = BaselineModel()
    model = AttnAggregateModel(3, args.adjust_weight, trained_baseline_model=trained_baseline_model, transform=args.transform)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_folder = str(config.TRAINED_MODELS / args.model_file_name)
    model_path = get_model_path(model_folder)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # evaluate train set
    train_data = json_load(config.FGC_TRAIN)
    train_set = AttnDataset(train_data, args.data_type, False, 3, 512, False)
    eval(model, train_data, args.batch_size, train_set, args.model_file_name)
    
    # evaluate dev set
    dev_data = json_load(config.FGC_DEV)
    dev_set = AttnDataset(dev_data, args.data_type, False, 3, 512, False)
    eval(model, dev_data, args.batch_size, dev_set, args.model_file_name)
    
    # evaluate test set
    test_data = json_load(config.FGC_TEST)
    test_set = AttnDataset(test_data, args.data_type, False, 3, 512, False)
    eval(model, test_data, args.batch_size, test_set, args.model_file_name)


def eval(model, data, batch_size, dataset, model_file_path):
    model.eval()
    cumulative_len = dataset.cumulative_len
    indices_golds = dataset.shints
    
    with torch.no_grad():
        counter = 0
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=False, num_workers=batch_size)
        indices_preds = []
        current_document_labels = []
        weights = []
        for batch in tqdm(dataloader):
            for key in batch.keys():
                batch[key] = batch[key].to(self.device)
            predict_se = self.model.module.predict if hasattr(self.model,
                                                              'module') else self.model.predict
            
            if isinstance(self.model.module, AttnAggregateModel):
                weight, current_labels = predict_se(batch)
                weights.append(weight)
            else:
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
        
        logger.debug("indices_golds:{}".format(len(indices_golds)))
        logger.debug("indices_preds:{}".format(len(indices_preds)))
        
        preprocessed_data = eval_preprocess(data, indices_preds)
        with open(model_file_path+'/analysis.txt', 'w') as f:
            f.write(get_analysis(preprocessed_data))
        
    metrics = eval_sp(indices_golds, indices_preds)
    logger.debug(indices_golds)
    logger.debug(indices_preds)
    
    print(weights)
    return metrics

def eval_preprocess(data, indices_preds):
    for document_i, document in enumerate(data):
        document['QUESTIONS'][0]['sp'] = indices_preds[document_i]
    return data
