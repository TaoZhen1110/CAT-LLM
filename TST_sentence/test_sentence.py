import argparse
import os
from openpyxl import Workbook
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer
from sentence_prepare import Textdataset, MyDataLoader

from Model import TextClassification

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-gpu', type=str, default='0,1,2')

    ##############   pretrained model setting
    parser.add_argument("-checkpoint", type=str, default='../pretrained',
                        help='主干网络检查点：hfl/chinese-macbert-base, bert-base-chinese, hfl/chinese-roberta-wwm-ext')
    parser.add_argument("-freeze", type=str, default="3",
                        help='冻结主干网络模式。0：不冻结；1：冻结word embedding；2：冻结全部embeddings；3：encoder只解冻pooler')
    parser.add_argument("-weight_path", type=str, default="./run_text/run_0/Share_text_best.pth")
    return parser.parse_args()

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    ###################  Dataset prepare  #################
    tokenizer = BertTokenizer.from_pretrained(args.checkpoint)
    testloader = MyDataLoader(dataset=Textdataset(branch="val", tokenizer=tokenizer), batch_size=1, shuffle=False)
    num_iter_ts = len(testloader)
    print(num_iter_ts)



if __name__ == '__main__':
    args = get_arguments()
    main(args)