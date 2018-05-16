import glob, os
from PIL import Image
from Params import *
files = glob.glob(os.path.join(DATA_DIR, "train_masks/*.gif"))

for imageFile in files:
    filepath,filename = os.path.split(imageFile)
    filterame,exts = os.path.splitext(filename)
    print "Processing: " + imageFile,filterame
    im = Image.open(imageFile)
    path = os.path.join(DATA_DIR, "train_masks_png", filename.split('.')[0]+'.png')
    #print path
    try:
    	im.save(path,'PNG')	
    except:
    	print path
    	print "fail!!"