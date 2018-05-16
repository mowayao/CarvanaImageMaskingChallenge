import cv2
import numpy as np
import pandas as pd
import threading
import queue
from tqdm import tqdm
from data import ImgData
import torch.utils.data as data_utils
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data.sampler import SequentialSampler
from Params import *
import os
df_test = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
ids_test = df_test['img'].map(lambda s: s.split('.')[0])

names = []
for id in ids_test:
    names.append('{}.jpg'.format(id))


MODEL.load_state_dict(torch.load('unet1024_2017-08-30+15%3A51%3A37.p'))
MODEL.cuda().eval()


print('Predicting on {} samples with batch_size = {}...'.format(len(ids_test), BATCH_SIZE))
for start in range(0, len(ids_test), BATCH_SIZE):
    x_batch = []
    end = min(start + BATCH_SIZE, len(ids_test))
    ids_test_batch = ids_test[start:end]
    for id in ids_test_batch.values:
        img = cv2.imread(os.path.join(DATA_DIR, 'test/{}.jpg'.format(id)))
        img = cv2.resize(img, INPUT_SIZE)
        img = np.transpose(img, [2,0,1])
        x_batch.append(img)
    x_batch = np.array(x_batch, np.float) / 255
    x_batch = torch.FloatTensor(x_batch)
    x_batch = Variable(x_batch).cuda()
    preds = MODEL(x_batch)
    preds = F.sigmoid(preds)
    preds = preds.cpu().data.numpy()
    preds = np.squeeze(preds, axis=1)
    for pred in preds:
        prob = cv2.resize(pred, ORIG_SIZE)
        mask = np.array(prob > THRESHOLD, np.float)
        cv2.imshow("pred", mask)
        cv2.waitKey(0)
