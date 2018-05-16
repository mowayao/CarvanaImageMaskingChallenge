import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import cv2
import torch.nn.functional as F
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from data import ImgMaskData
from Params import *
import torch
from torch.utils.data.sampler import SequentialSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
class BCELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs        = F.sigmoid(logits)
        probs_flat   = probs.view (-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)
def criterion(logits, labels):
    loss = BCELoss2d()(logits, labels)  + DiceLoss()(logits, labels)
    return loss

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
    def forward(self, logits, targets):
        num = targets.size(0)
        probs = F.sigmoid(logits)
        m1  = probs.view(num,-1)
        m2  = targets.view(num,-1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
        score = 1- score.sum()/num
        return score


def train():

    path = os.path.join(DATA_DIR, 'train_masks.csv')
    df_train = pd.read_csv(path)
    ids_train = df_train['img'].map(lambda s: s.split('.')[0])
    ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=SPLIT_RATIO, random_state=SEED)
    train_data_folder = ImgMaskData(ids_train_split, INPUT_SIZE)
    val_data_folder = ImgMaskData(ids_valid_split, INPUT_SIZE)
    train_data_loader = data_utils.DataLoader(train_data_folder, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    #val_data_loader = data_utils.DataLoader(val_data_folder, sampler=SequentialSampler(val_data_folder), batch_size=BATCH_SIZE, num_workers=2, drop_last=False)
    MODEL.cuda()
    dloss = DiceLoss()
    bce_loss = BCELoss2d()
    learning_rate = 0.01
    optimizer = optim.Adam(MODEL.parameters(), lr=learning_rate)
    total_iter = 0.
    for epoch in xrange(EPOCHS):
        loss_aver = 0.
        loss_bce_aver = 0.
        loss_dice_aver = 0.
        MODEL.train()
        for iter_num, (img_batch, mask_batch) in enumerate(train_data_loader):
            total_iter += 1
            img_batch = Variable(img_batch).cuda()
            mask_batch = Variable(mask_batch).cuda()
            pred_mask_batch = MODEL(img_batch)
            loss = criterion(pred_mask_batch, mask_batch)
            dice_loss = dloss(pred_mask_batch, mask_batch)
            BCE_loss = bce_loss(pred_mask_batch, mask_batch)
            loss_aver += loss.cpu().data.numpy()[0]
            loss_dice_aver += dice_loss.cpu().data.numpy()[0]
            loss_bce_aver += BCE_loss.cpu().data.numpy()[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learning_rate = learning_rate * ((1.0 - total_iter / (EPOCHS * len(train_data_folder) / BATCH_SIZE))**0.9)
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            if (iter_num+1) % LOG_INTERVAL == 0:
                print('Train Iteration: {}/{} \tLoss: {:.6f} \t bce loss: {}\t dice loss: {}\t lr:{}\t'.format(
                    iter_num+1,
                    len(train_data_folder)/BATCH_SIZE,
                    loss_aver/(iter_num+1),
                    loss_bce_aver/(iter_num+1),
                    loss_dice_aver/(iter_num+1),
                    learning_rate
                ))

                #lr_scheduler.step(loss_aver/(iter_num+1))
    torch.save(MODEL.state_dict(), "unet512.p")
if __name__ == "__main__":
    train()