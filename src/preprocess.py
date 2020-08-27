import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import *
from tqdm import tqdm

from .std import *

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
device = torch.device("cuda")
class AttnExample:
    """
    A single instance for attn_aggregate model.
    """

    def __init__(self, documentSentences, shint, question_str: str, sentences: List[str], sentence_mask: List[int], label, multiBERTs):
        """

        :param question_str:
        :param sentences: list of string, the length is sentence window. ["[PAD]", ..., ...]
        :param sentence_mask: indicate the padding sentence. e.g., [1, 0, 0]
        :param label: the label of the middle sentence.
        """
        self.documentSentences = documentSentences
        self.shint = shint
        self.question = question_str
        self.sentences = sentences
        self.sentence_mask = sentence_mask
        self.label = label
        self.multiBERTs = multiBERTs
        
    def convert2tensor(self):
        qa_pairs = []

        for s in self.sentences:
            qa_pairs.append([self.question, s])

        tensor_inp = tokenizer(qa_pairs, padding='max_length', truncation='longest_first', max_length=300,
                              return_tensors='pt')

        if self.multiBERTs:
            ss_pairs = []
            for s in self.sentences:
                ss_pairs.append([self.sentences[len(self.sentences) // 2], s])
            tensor_inp_ss = tokenizer(ss_pairs, padding='max_length', truncation='longest_first', max_length=300, return_tensors='pt')


            tensor_inp['nsp_input_ids'] = tensor_inp_ss['input_ids']
            tensor_inp['nsp_attention_mask'] = tensor_inp_ss['attention_mask']
            tensor_inp['nsp_token_type_ids'] = tensor_inp_ss['token_type_ids']
            
        tensor_inp['label'] = torch.tensor(self.label)
        tensor_inp['sentence_mask'] = torch.tensor(self.sentence_mask)
        #tensor_inp['input_ids'] = tensor_inp['input_ids'].to(device)
        #tensor_inp['attention_mask'] = tensor_inp['attention_mask'].to(device)
        #tensor_inp['token_type_ids'] = tensor_inp['token_type_ids'].to(device)
        #tensor_inp['label'] = tensor_inp['label'].to(device)

        return tensor_inp

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances[idx]
        return instance

def data_preprocessing(data, sentence_window, multiBERTs):
    out_examples = []
    cumulative_len = []
    for document in tqdm(data):
        question = document['QUESTIONS'][0]
        question_str= question['QTEXT_CN']
        shint = question['SHINT_']
        if not shint: continue # eliminate question with no SE
        documentSentences = [s['text'] for s in document['SENTS']]
        cumulative_len.append(len(documentSentences))
        for s_i, s in enumerate(documentSentences):
            sentences = []
            sentence_masks = []
            for j in range(s_i - sentence_window // 2, s_i + sentence_window // 2 + 1):
                if j < 0 or j >= len(documentSentences):
                    sentences.append('[PAD]')
                    sentence_masks.append([0])
                else:
                    sentences.append(documentSentences[j])
                    sentence_masks.append([1])

            if s_i in shint:
                label = [1]
            else:
                label = [0]

            out_examples.append(AttnExample(documentSentences, shint, question_str, sentences, sentence_masks, label, multiBERTs))
            
    cumulative_len = np.cumsum(cumulative_len).tolist()
    
    return out_examples, cumulative_len

def data_preprocessing_ssqa(data, sentence_window, multiBERTs):
    out_examples = []
    cumulative_len = []
    for document in tqdm(data):
        question = document['qtext']
        shint = document['se']
        if not shint: continue # eliminate question with no SE
        documentSentences = [[s[2], p_i, s_i] for p_i, p in enumerate(train_data[0]['context']) for s_i, s in enumerate(p[2])]
        cumulative_len.append(len(documentSentences))
        for s_i, s in enumerate(documentSentences):
            sentences = []
            sentence_masks = []
            for j in range(s_i - sentence_window // 2, s_i + sentence_window // 2 + 1):
                if j < 0 or j >= len(documentSentences):
                    sentences.append('[PAD]')
                    sentence_masks.append([0])
                else:
                    sentences.append(documentSentences[j][0])
                    sentence_masks.append([1])
            
            current_shint = [s[1], s[2]]
            if current_shint in shint:
                label = [1]
            else:
                label = [0]

            out_examples.append(AttnExample(documentSentences, shint, question, sentences, sentence_masks, label, multiBERTs))
            
    cumulative_len = np.cumsum(cumulative_len).tolist()
    
    return out_examples, cumulative_len

class AttnDataset(Dataset):
    def __init__(self, data, data_type, multiBERTs, number_of_sentences):
        """
        :param json_fp: e.g, "FGC_release_1.7.13/FGC_release_all_train.json"
        """
        self.instances = []
        self.data_type = data_type
        
        if data_type == 'fgc':
            examples, cumulative_len = data_preprocessing(data, number_of_sentences, multiBERTs)
        if data_type == 'ssqa':
            examples, cumulative_len = data_preprocessing_ssqa(data, number_of_sentences, multiBERTs)
        
        self.cumulative_len = cumulative_len
        self.shints = []
        
        for e in examples:
            self.instances.append(e.convert2tensor())
            
        if data_type == 'fgc':
            for document in data:
                shint = document['QUESTIONS'][0]['SHINT_']
                if shint:
                    self.shints.append(shint)
        if data_type == 'ssqa':
            for document in data:
                shint = document['se']
                if shint:
                    self.shints.append(shint)
                    
        print(len(self.shints))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances[idx]
        return instance
    
    def __getdata__(self, data):
        return data