import torch
from torch.utils.data import Dataset
from typing import *

from .std import *

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

class AttnExample:
    """
    A single instance for attn_aggregate model.
    """

    def __init__(self, documentSentences, question_str: str, sentences: List[str], sentence_mask: List[int], label):
        """

        :param question_str:
        :param sentences: list of string, the length is sentence window. ["[PAD]", ..., ...]
        :param sentence_mask: indicate the padding sentence. e.g., [1, 0, 0]
        :param label: the label of the middle sentence.
        """
        self.documentSentences = documentSentences
        self.question = question_str
        self.sentences = sentences
        self.sentence_mask = sentence_mask
        self.label = label

    def convert2tensor(self):
        qa_pairs = []
        for s in self.sentences:
            qa_pairs.append([self.question, s])

        tensor_inp = tokenizer(qa_pairs, padding='max_length', truncation='longest_first', max_length=300,
                              return_tensors='pt')

        tensor_inp['label'] = torch.tensor(self.label)

        return tensor_inp


def data_preprocessing(data, sentence_window):
    out_examples = []
    for document in data:
        question = document['QUESTIONS'][0]
        question_str= question['QTEXT_CN']
        shint =  question['SHINT_']
        if not shint: continue # eliminate question with no SE
        documentSentences = [s['text'] for s in document['SENTS']]
        for s_i, s in enumerate(documentSentences):
            documentSentences.append(documentSentences[s_i])
            sentences = []
            sentence_masks = []
            for j in range(s_i - sentence_window // 2, s_i + sentence_window // 2 + 2):
                if j < 0 or j >= len(documentSentences):
                    sentences.append('[PAD]')
                    sentence_masks.append(1)
                else:
                    sentences.append(documentSentences[s_i])
                    sentence_masks.append(0)

            if s_i in shint:
                label = 1
            else:
                label = 0

            out_examples.append(AttnExample(documentSentences, question_str, sentences, sentence_masks, label))

    return out_examples


class AttnDataset(Dataset):
    def __init__(self, json_fp):
        """
        :param json_fp: e.g, "FGC_release_1.7.13/FGC_release_all_train.json"
        """
        self.instances = []
        data = json_load(json_fp)
        examples: List[AttnExample] = data_preprocessing(data, 3)

        for e in examples:
            self.instances.append(e.convert2tensor())

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances[idx]
        return instance