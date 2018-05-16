DATA_DIR = "/media/mowayao/yao_data/CarvanaImageMaskingChallenge"
BATCH_SIZE = 3
NUM_GRAD_ACC = 15//BATCH_SIZE
INPUT_SIZE = (1280, 1280)
SPLIT_RATIO = 0.1
SEED = 66
LOG_INTERVAL = 20
EPOCHS = 45
from models.unet import UNet1024
ORIG_SIZE = (1918, 1280)
THRESHOLD = 0.5
MODEL = UNet1024(1)
MODEL.cuda()