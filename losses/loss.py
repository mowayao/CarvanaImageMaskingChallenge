
import torch
import torch.nn as nn
import torch.nn.functional as F

# https://stackoverflow.com/questions/45184741/bceloss-for-binary-pixel-wise-segmentation-pytorch
# https://github.com/pytorch/pytorch/issues/751
# for formula: http://geek.csdn.net/news/detail/126833
# class StableBCELoss(nn.modules.Module):
#        def __init__(self):
#              super(StableBCELoss, self).__init__()
#        def forward(self, logit, label):
#              neg_abs = - logit.abs()
#              loss = logit.clamp(min=0) - logit * label + (1 + neg_abs.exp()).log()
#              return loss.mean()


class BCELoss2d(nn.Module):
    def __init__(self):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        #self.bce_loss = StableBCELoss()
    def forward(self, logits, labels):
        logits_flat = logits.view (-1)
        labels_flat = labels.view(-1)
        return self.bce_loss(logits_flat, labels_flat)


##  http://geek.csdn.net/news/detail/126833
class WeightedBCELoss2d(nn.Module):
    def __init__(self):
        super(WeightedBCELoss2d, self).__init__()

    def forward(self, logits, labels, weights):
        w = weights.view(-1)
        z = logits.view (-1)
        t = labels.view (-1)
        loss = w*z.clamp(min=0) - w*z*t + w*torch.log(1 + torch.exp(-z.abs()))
        loss = loss.sum()/w.sum()
        return loss



class WeightedSoftDiceLoss(nn.Module):
    def __init__(self):
        super(WeightedSoftDiceLoss, self).__init__()

    def forward(self, logits, labels, weights):
        probs = F.sigmoid(logits)
        num   = labels.size(0)
        w     = (weights).view(num,-1)
        w2    = w*w
        m1    = (probs  ).view(num,-1)
        m2    = (labels ).view(num,-1)
        intersection = (m1 * m2)
        score = 2. * ((w2*intersection).sum(1)+1) / ((w2*m1).sum(1) + (w2*m2).sum(1)+1)
        score = 1 - score.sum()/num
        return score

class SoftDiceLoss(nn.Module):
    def __init__(self):  #weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, labels):
        probs = F.sigmoid(logits)
        num = labels.size(0)
        m1  = probs.view (num,-1)
        m2  = labels.view(num,-1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
        score = 1- score.sum()/num
        return score


class DiceAccuracy(nn.Module):
    def __init__(self):  #weight=None, size_average=True):
        super(DiceAccuracy, self).__init__()

    def forward(self, probs, labels):
        num = labels.size(0)
        m1  = probs.view (num,-1)
        m2  = labels.view(num,-1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
        score = score.sum()/num
        return score


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()


    def forward(self, logits, labels):
        probs = F.sigmoid(logits)
        probs_flat  = probs.view (-1)
        labels_flat = labels.view(-1)
        loss = torch.mean((probs_flat - labels_flat) ** 2)
        return loss

# helper-------------------------------------------------------------------------------------------
#https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
#https://www.kaggle.com/c/carvana-image-masking-challenge#evaluation

#https://github.com/pytorch/pytorch/issues/1249
def dice_loss(m1, m2, is_average=True):
    num = m1.size(0)
    m1  = m1.view(num,-1)
    m2  = m2.view(num,-1)
    intersection = (m1 * m2)
    scores = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
    if is_average:
        score = scores.sum()/num
        return score
    else:
        return scores


def one_dice_loss_py(m1, m2):
    m1 = m1.reshape(-1)
    m2 = m2.reshape(-1)
    intersection = (m1 * m2)
    score = 2. * (intersection.sum()+1) / (m1.sum() + m2.sum()+1)
    return score


