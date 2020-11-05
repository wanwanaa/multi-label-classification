import torch
import numpy as np
from pytorch_pretrained_bert import BertTokenizer

class Datasets():
    def __init__(self, config):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    def _covert_bert(self, line):
        return ["[CLS]"] + line + ["[SEP]"]
    
    def _generate_vocab(self, file):
        vocab = {}
        with open(file, 'r') as f:
            for line in f:
                line = line.strip().split()
                vocab[line[0]] = line[1]
        return vocab

    def _get_txtset(self, file_txt, file_label):
        txts = []
        labels = []
        with open(file_txt, 'r') as f_txt, open(file_label, 'r') as f_label:
            for (txt, label) in zip(file_txt, f_label):
                txts.append(txt.strip())
                labels(label.strip().split())
        return txts, labels

    def get_dataset(self, file_txt, file_label, file_vocab):
        vocab = _generate_vocab(file_vocab)
        txts, labels = _get_txtset(file_txt, file_label, file_vocab)
        dataset_labels = np.zeros([len(labels), self.config.multi_num])
        dataset_txts = np.zeros([len(txts), self.max_seq_len])
        for step, (txt, label) in enumerate(zip(txts, labels)):
            # label
            dataset_labels[i] = [vocab[l] for l in label]

            # txt
            line = self.tokenizer.tokenize(txts)
            line = self._covert_bert(ids)
            ids = tokenizer.convert_tokens_to_ids(line)
            if len(ids) <= max_length:
                ids = np.pad(np.array(ids), (0, self.config.max_seq_len-len(ids), "constant")
            else:
                ids = ids[:self.config.max_seq_len]
            dataset_txts[i] = ids
        dataset_labels = torch.form_numpy(dataset_labels).type(torch.int32)
        dataset_txts = torch.form_numpy(dataset_txts).type(torch.int32)
        return dataset_txts, dataset_labels

