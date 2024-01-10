import os
import sys
import torch
from torch.utils.data import Dataset
import pandas as pd

class Textdataset(Dataset):
    def __init__(self, branch = "train", tokenizer=None):
        super(Textdataset,self).__init__()
        # df = pd.read_excel("./text_dataset/%s.tsv" % branch, encoding='utf-8', sep="\t").fillna('')
        # df = pd.read_excel("/home/taozhen/sentiment_analysis_master/TST_sentence/data_set/整散句_%s.xlsx" % branch,
        #                    header=None)
        df = pd.read_excel("/home/taozhen/sentiment_analysis_master/TST_sentence/data_set/修辞分类_%s.xlsx" % branch,
                                              header=None)
        self.tokenizer = tokenizer
        self.data = df.values.tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        enc_input = self.tokenizer(
            self.data[idx][1],
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return enc_input.input_ids.squeeze(), enc_input.attention_mask.squeeze(), \
               enc_input.token_type_ids.squeeze(), \
               self.data[idx][0]

class sentencedata(Dataset):
    def __init__(self, sentences, tokenizer=None):
        super(sentencedata, self).__init__()
        self.tokenizer = tokenizer
        self.data = sentences
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        enc_input = self.tokenizer(
            self.data[idx],
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return enc_input.input_ids.squeeze(), enc_input.attention_mask.squeeze(), \
            enc_input.token_type_ids.squeeze()



def MyDataLoader(dataset, batch_size, shuffle=False):
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                              shuffle=shuffle, num_workers=nw)
    return data_loader

