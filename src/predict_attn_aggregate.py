from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from .preprocess import AttnDataset
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
    parser.add_argument('-sentence',
                        type=int,
                        default=1)
    parser.add_argument('-multiBERTs',
                        default=False,
                        action='store_true')
    parser.add_argument('-max_length',
                        type=int,
                        default=512)
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
    model = AttnAggregateModel(args.sentence, args.adjust_weight, trained_baseline_model=trained_baseline_model, transform=args.transform)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_folder = str(config.TRAINED_MODELS / args.model_file_name)
    model_path = get_model_path(model_folder)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # train set
    logger.info("Evaluate train set")
    train_data = json_load(config.FGC_TRAIN)
    train_set = AttnDataset(train_data, args.data_type, args.multiBERTs, args.sentence, args.max_length, False)
    get_eval(model, train_data, args.batch_size, train_set, args.model_file_name, device, "train_analysis.txt")
    
    # dev set
    logger.info("Evaluate dev set")
    dev_data = json_load(config.FGC_DEV)
    dev_set = AttnDataset(dev_data, args.data_type, args.multiBERTs, args.sentence, args.max_length, False)
    get_eval(model, dev_data, args.batch_size, dev_set, args.model_file_name, device, "dev_analysis.txt")
    
    # test set
    logger.info("Evaluate test set")
    test_data = json_load(config.FGC_TEST)
    test_set = AttnDataset(test_data, args.data_type, args.multiBERTs, args.sentence, args.max_length, False)
    get_eval(model, test_data, args.batch_size, test_set, args.model_file_name, device, "test_analysis.txt")


def get_eval(model, data, batch_size, dataset, model_file_path, device, output_file_name):
    cumulative_len = dataset.cumulative_len
    indices_golds = dataset.shints
    
    with torch.no_grad():
        counter = 0
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=False, num_workers=batch_size)
        indices_preds = []
        all_weights = []
        all_scores = []

        current_document_labels = []
        current_document_weights = []
        current_document_scores = []
        for batch in tqdm(dataloader):
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            predict_se = model.module.predict if hasattr(model, 'module') else model.predict
            current_labels, current_scores, current_weights = predict_se(batch)

            for label, weight, score in zip(current_labels, current_weights, current_scores):
                if counter + 1 in cumulative_len:
                    current_document_labels += label
                    current_document_weights.append(weight)
                    current_document_scores += score
                    current_document_indices = np.where(np.array(current_document_labels) == 1)[0].tolist()
                    indices_preds.append(current_document_indices)
                    all_weights.append(current_weights)
                    all_scores.append(current_document_scores)
                    current_document_labels = []
                    current_document_weights = []
                    current_document_scores = []
                
                else:
                    current_document_labels += label
                    current_document_weights.append(weight)
                    current_document_scores += score
                
                counter = counter + 1
        
    logger.debug("indices_golds:{}".format(len(indices_golds)))
    logger.debug("indices_preds:{}".format(len(indices_preds)))

    assert len(indices_golds) == len(indices_preds) == len(all_weights)

    metrics = eval_sp(indices_golds, indices_preds)
    preprocessed_data = eval_preprocess(data, indices_preds, all_weights, all_scores)
    with open(model_file_path+'/'+output_file_name, 'w') as f:
        f.write(str(metrics) + '\n')
        f.write(get_analysis(preprocessed_data))


def eval_preprocess(data, indices_preds, all_weights, all_scores):
    for document_i, document in enumerate(data):
        document['QUESTIONS'][0]['sp'] = indices_preds[document_i]
        document['QUESTIONS'][0]['scores'] = all_scores[document_i]
        document['QUESTIONS'][0]['weights'] = all_weights[document_i]
    return data


if __name__=='__main__':
    main()

