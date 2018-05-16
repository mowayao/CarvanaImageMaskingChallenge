import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import torch.nn.functional as F
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from data import ImgMaskData
from Params import *
import torch
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from losses.loss import WeightedBCELoss2d, WeightedSoftDiceLoss, dice_loss
import time

#import torch.nn.utils.clip_grad_norm as clip_grad
import datetime
import logging

def criterion(logits, labels, is_weight=True):
    # compute weights
    btach_size, H, W = labels.size()
    if H == 128:
        kernel_size = 11
    elif H == 256:
        kernel_size = 21
    elif H == 512:
        kernel_size = 21
    elif H == 1024:
        kernel_size = 41
    elif H == 1280:
        kernel_size = 61
    else:
        raise ValueError('exit at criterion()')
    a = F.avg_pool2d(labels, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
    ind = a.ge(0.01) * a.le(0.99)
    ind = ind.float()
    weights = Variable(torch.tensor.torch.ones(a.size())).cuda()
    if is_weight:
        w0 = weights.sum()
        weights = weights + ind * 2
        w1 = weights.sum()
        weights = weights / w1 * w0
    return WeightedBCELoss2d()(logits, labels, weights) + \
        WeightedSoftDiceLoss()(logits, labels, weights)

def evaluate(val_data):
    MODEL.eval()
    test_acc = 0
    test_loss = 0
    test_num = 0
    for iter_num, (img_batch, label_batch) in enumerate(val_data):
        img_batch = Variable(img_batch.cuda(), volatile=True)
        label_batch = Variable(label_batch.cuda(), volatile=True)

        # forward
        logits = MODEL(img_batch)
        probs = F.sigmoid(logits)
        masks = (probs > 0.5).float()

        loss = criterion(logits, label_batch)
        acc = dice_loss(masks, label_batch)

        batch_size = label_batch.size(0)
        test_num += batch_size
        test_loss += batch_size * loss.data[0]
        test_acc += batch_size * acc.data[0]
    test_loss = test_loss / test_num
    test_acc = test_acc / test_num

    return test_loss, test_acc


def train():

    path = os.path.join(DATA_DIR,'train_masks.csv')
    df_train = pd.read_csv(path)
    ids_train = df_train['img'].map(lambda s: s.split('.')[0])
    ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=SPLIT_RATIO, random_state=SEED)
    train_data_folder = ImgMaskData(ids_train_split, INPUT_SIZE)
    val_data_folder = ImgMaskData(ids_valid_split, INPUT_SIZE, False)
    train_data_loader = data_utils.DataLoader(train_data_folder, batch_size=BATCH_SIZE, sampler=RandomSampler(train_data_folder), num_workers=4,
                                              drop_last=True)
    val_data_loader = data_utils.DataLoader(val_data_folder, batch_size=BATCH_SIZE, sampler=SequentialSampler(val_data_folder), num_workers=4,
                                            drop_last=False)


    date_time = datetime.datetime.fromtimestamp(
        int(time.time())
    ).strftime('%Y-%m-%d-%H-%M-%S')
    logging.basicConfig(filename='logs/{}.log'.format(date_time),level=logging.DEBUG)
    weights_path_name = "weights/{}".format(date_time)
    checkpoint_path_name = "checkpoint/{}".format(date_time)
    if not os.path.exists(weights_path_name):
        os.mkdir(weights_path_name)
    if not os.path.exists(checkpoint_path_name):
        os.mkdir(checkpoint_path_name)
    op = optim.Adam(MODEL.parameters(), lr=1e-3, weight_decay=5e-4)
    optim.RMSprop
    optimizer = optim.SGD(MODEL.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    scheduler = MultiStepLR(optimizer, milestones=[35, 40, 50, 55], gamma=0.5)
    optimizer.zero_grad()
    total_iter = 0.
    epoch_save = list(range(1, EPOCHS + 1, 3))
    for epoch in xrange(EPOCHS):
        scheduler.step()
        print "Epoch:{}/{}".format(epoch+1, EPOCHS)
        sum_train_loss = 0.0
        sum_train_acc = 0.0
        cnt = 0
        if epoch in epoch_save:
            torch.save(MODEL.state_dict(), weights_path_name + '/%03d.pth' % epoch)
            torch.save({
                'state_dict': MODEL.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }, checkpoint_path_name+"/%03d.pth" % epoch)
        MODEL.train()
        optimizer.zero_grad()
        for iter_num, (img_batch, label_batch) in enumerate(train_data_loader):
            total_iter += 1
            img_batch = Variable(img_batch).cuda()
            label_batch = Variable(label_batch).cuda()
            logits = MODEL(img_batch)
            probs = F.sigmoid(logits)
            loss = criterion(logits, label_batch) / NUM_GRAD_ACC
            masks = (probs > 0.5).float()
            acc = dice_loss(masks, label_batch)
            sum_train_acc += acc.data[0]
            sum_train_loss += loss.data[0] * NUM_GRAD_ACC
            cnt += 1
            loss.backward()##accumlate gradients

            if (iter_num + 1) % NUM_GRAD_ACC == 0:
                optimizer.step()
                optimizer.zero_grad()
            if (iter_num + 1) % LOG_INTERVAL == 0:
                print('Train Iteration: {}/{} \tLoss: {:.6f} \t Accuracy: {:.6f}\t'.format(
                    iter_num + 1,
                    len(train_data_folder) / BATCH_SIZE,
                    sum_train_loss / cnt,
                    sum_train_acc / cnt,
                ))
                sum_train_acc = 0.
                sum_train_loss = 0.
                cnt = 0

        test_loss, test_acc = evaluate(val_data_loader)
        logging.info('Validation: \tLoss: {:.6f} \t Accuracy: {:.6f}\t'.format(
            test_loss,
            test_acc
        ))
        print('Validation: \tLoss: {:.6f} \t Accuracy: {:.6f}\t'.format(
            test_loss,
            test_acc
        ))

if __name__ == "__main__":
    train()



