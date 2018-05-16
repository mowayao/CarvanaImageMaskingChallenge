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
from torch.utils.data.sampler import SequentialSampler
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from losses.loss import WeightedBCELoss2d, WeightedSoftDiceLoss

def criterion(logits, labels, w=(1, 1), is_weight=True, ):
    a = F.avg_pool2d(labels, kernel_size=11, padding=5, stride=1)
    ind = a.ge(0.01) * a.le(0.99)
    ind = ind.float()
    weights = Variable(torch.ones(a.size())).cuda()
    if is_weight:
        w0 = weights.sum()
        weights = weights + ind*2
        w1 = weights.sum()
        weights = weights/w1*w0
    bce_loss = WeightedBCELoss2d()(logits, labels, weights)
    dice_loss = WeightedSoftDiceLoss()(logits, labels, weights)
    return w[0] * bce_loss + w[1] * dice_loss, bce_loss, dice_loss



def train():

    path = os.path.join(DATA_DIR,'train_masks.csv')
    df_train = pd.read_csv(path)
    ids_train = df_train['img'].map(lambda s: s.split('.')[0])
    ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=SPLIT_RATIO, random_state=SEED)
    train_data_folder = ImgMaskData(ids_train_split, INPUT_SIZE)
    val_data_folder = ImgMaskData(ids_valid_split, INPUT_SIZE, False)
    train_data_loader = data_utils.DataLoader(train_data_folder, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
                                              drop_last=True)
    val_data_loader = data_utils.DataLoader(val_data_folder, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                                            drop_last=False)


    learning_rate = 0.01

    optimizer = optim.RMSprop(MODEL.parameters(), lr=learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=[15, 30], gamma=0.5)
    optimizer.zero_grad()
    total_iter = 0.
    for epoch in xrange(EPOCHS):
        scheduler.step()
        print "Epoch:{}/{}".format(epoch+1, EPOCHS)
        loss_aver = 0.
        loss_bce_aver = 0.
        loss_dice_aver = 0.
        MODEL.train()
        optimizer.zero_grad()
        for iter_num, (img_batch, mask_batch) in enumerate(train_data_loader):
            total_iter += 1
            img_batch = Variable(img_batch).cuda()
            mask_batch = Variable(mask_batch).cuda()
            pred_mask_batch = MODEL(img_batch)
            loss, bce_loss, dice_loss = criterion(pred_mask_batch, mask_batch)
            loss_aver += loss.cpu().data.numpy()[0]
            loss_dice_aver += dice_loss.cpu().data.numpy()[0]
            loss_bce_aver += bce_loss.cpu().data.numpy()[0]

            loss.backward()##accumlate gradients

            if (iter_num + 1) % LOG_INTERVAL == 0:
                optimizer.step()
                optimizer.zero_grad()
                print('Train Iteration: {}/{} \tLoss: {:.6f} \t bce loss: {:.6f}\t dice loss: {:.6f}\t'.format(
                    iter_num + 1,
                    len(train_data_folder) / BATCH_SIZE,
                    loss_aver/(iter_num+1),
                    loss_bce_aver/(iter_num+1),
                    loss_dice_aver/(iter_num+1)
                ))

        MODEL.eval()
        loss_aver = 0.
        loss_bce_aver = 0.
        loss_dice_aver = 0.
        cnt = 0
        for iter_num, (img_batch, mask_batch) in enumerate(val_data_loader):
            img_batch = Variable(img_batch).cuda()
            mask_batch = Variable(mask_batch).cuda()
            pred_mask_batch = MODEL(img_batch)
            loss, bce_loss, dice_loss = criterion(pred_mask_batch, mask_batch)
            loss_aver += loss.cpu().data.numpy()[0]
            loss_dice_aver += dice_loss.cpu().data.numpy()[0]
            loss_bce_aver += bce_loss.cpu().data.numpy()[0]
            cnt += 1
        print('Validate Epoch: {}/{} \tLoss: {:.6f} \t bce loss: {}\t dice loss: {}\t'.format(
            epoch,
            EPOCHS,
            loss_aver/cnt,
            loss_bce_aver/cnt,
            loss_dice_aver/cnt
        ))

    torch.save(MODEL.state_dict(), "unet1024.p")
if __name__ == "__main__":
    train()



