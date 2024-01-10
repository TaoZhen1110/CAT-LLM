import argparse   #用于解析命令行参数
import glob       #查找文件目录和文件
import os         #提供的就是各种 Python 程序与操作系统进行交互的接口
import random     #用于生成随机数
import socket
import time      #用于格式化日期和时间
from datetime import datetime
from tqdm import tqdm     #python进度条库
import numpy as np      #支持大量的维度数组与矩阵运算

from Model import TextClassification

# PyTorch includes
import torch
import torch.nn as nn

# Tensorboard include
from tensorboardX import SummaryWriter

# Dataloaders includes
from transformers import BertTokenizer
from sentence_prepare import Textdataset, MyDataLoader


def get_arguments():
    parser = argparse.ArgumentParser()   #创建解析器,ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息

    parser.add_argument('-gpu', type=str, default='0,1,2')
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-nepochs', type=int, default=100)
    parser.add_argument('-resume_epoch', type=int, default=0)
    parser.add_argument('-log_every', type=int, default=20)
    parser.add_argument('-naver_grad', type=int, default=1)

    ##############   pretrained model setting
    parser.add_argument("-checkpoint", type=str,default='../pretrained',
                        help='主干网络检查点：hfl/chinese-macbert-base, bert-base-chinese, hfl/chinese-roberta-wwm-ext')
    parser.add_argument("-freeze", type=str, default="0",
                        help='冻结主干网络模式。0：不冻结；1：冻结word embedding；2：冻结全部embeddings；3：encoder只解冻pooler')

    ##############   Optimizer settings
    parser.add_argument('-lr', type=float, default=2e-5, help='lr')

    return parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)          #CPU设置种子，生成随机数,number bianhao
    torch.cuda.manual_seed_all(seed)  #为所有GPU设置种子，生成随机数
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True   #每次返回的卷积算法将是确定的

def main(args):
    setup_seed(1234)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    #################  Train model save setting   ###################
    save_dir_root = os.path.dirname(os.path.abspath(__file__))
    # os.path.abspath:取当前文件的绝对路径（完整路径）
    # os.path.dirname:去掉文件名，返回目录

    if args.resume_epoch != 0:   #resume_epoch=0
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run_text', 'run_*')))
        #os.path.join:连接两个或更多的路径名组件
        run_id = int(runs[-1].split('_')[-1]) if runs else 0
    else:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run_text', 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0          #通过指定分隔符对字符串进行切片

    save_dir = os.path.join(save_dir_root, 'run_text', 'run_' + str(run_id))
    log_dir = os.path.join(save_dir, datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)  # 将数据以特定的格式存储到上面得到的那个日志文件夹中

    ###################  Dataset prepare #################
    tokenizer = BertTokenizer.from_pretrained(args.checkpoint)
    trainloader = MyDataLoader(dataset=Textdataset(branch="train", tokenizer=tokenizer),
                               batch_size=args.batch_size, shuffle=True)
    valloader = MyDataLoader(dataset=Textdataset(branch="val", tokenizer=tokenizer), batch_size=1, shuffle=False)
    num_iter_tr = len(trainloader)
    num_iter_ts = len(valloader)
    nitrs = args.resume_epoch * num_iter_tr
    nsamples = 0

    ##################  Model ###################
    model = TextClassification(checkpoint=args.checkpoint, freeze=args.freeze)
    model = nn.DataParallel(model).cuda()

    ##################  优化 Setting #################
    parameters_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters_update, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

    #################  Loss function  #################
    loss_function = nn.CrossEntropyLoss()

    ##################   Train Model  ###################
    sample_num = 0
    recent_losses = []
    aveGrad = 0
    best_f, cur_f = 0.0, 0.0
    start_time = time.time()

    for epoch in tqdm(range(args.resume_epoch, args.nepochs)):
        model.train()
        epoch_losses = []

        for step, sample_batched in enumerate(trainloader):
            input_ids, attention_mask, token_type_ids, labels = (x.cuda() for x in sample_batched)
            sample_num += input_ids.shape[0]
            outputs = model(enc_inputs=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = loss_function(outputs, labels)
            trainloss = loss.item()
            epoch_losses.append(trainloss)

            if len(recent_losses) < args.log_every:  # args.log_every=40
                recent_losses.append(trainloss)
            else:
                recent_losses[nitrs % len(recent_losses)] = trainloss

            # Backward the averaged gradient
            loss.backward()
            aveGrad += 1
            nitrs += 1
            nsamples += args.batch_size

            # Update the weights once in p['nAveGrad'] forward passes
            if aveGrad % args.naver_grad == 0:  # args.naver_grad =1
                optimizer.step()  # 这个方法会更新所有的参数
                optimizer.zero_grad()
                aveGrad = 0

            if nitrs % args.log_every == 0:  # log_every=40
                meanloss1 = sum(recent_losses) / len(recent_losses)
                print('epoch: %d step: %d trainloss: %.4f timecost:%.2f secs' %
                      (epoch, step, meanloss1, time.time() - start_time))
                writer.add_scalar('data/trainloss', meanloss1, nsamples)

        meanloss2 = sum(epoch_losses) / len(epoch_losses)
        print('epoch: %d meanloss: %.4f' % (epoch, meanloss2))
        writer.add_scalar('data/epochloss', meanloss2, nsamples)

        scheduler.step()

        ######################## eval model ###########################
        print("######## val data ########")

        val_ACC = 0.0
        # prec_lists = []
        # recall_lists = []
        sum_valloss = 0.0
        count = 0
        model.eval()

        for step, sample_batched in enumerate(valloader):
            input_ids, attention_mask, token_type_ids, label = (x.cuda() for x in sample_batched)
            with torch.no_grad():
                outputs = model(enc_inputs=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

            loss = loss_function(outputs, label)
            sum_valloss += loss.item()
            pre_label = torch.argmax(outputs, 1)

            count += 1
            val_ACC += torch.sum(pre_label == label)

            if step % num_iter_ts == num_iter_ts - 1:
                mean_valLoss = sum_valloss / num_iter_ts
                mean_valAcc = val_ACC / count

                print('Validation:')
                print('epoch: %d, numImages: %d valLoss: %.4f Accuracy: %.4f' % (
                    epoch, count, mean_valLoss, mean_valAcc))
                writer.add_scalar('data/valloss', mean_valLoss, count)
                writer.add_scalar('data/valAcc', mean_valAcc, count)

                ################   Save Pth  ################
                cur_f = mean_valAcc
                if cur_f > best_f:
                    save_path = os.path.join(save_dir, 'rhetoric' + '.pth')
                    torch.save(model.state_dict(), save_path)
                    print("Save model at {}\n".format(save_path))
                    best_f = cur_f

if __name__ == "__main__":
    args = get_arguments()
    main(args)