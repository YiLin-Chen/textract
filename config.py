# parameters for box extraction
BOX_DILATE_KERNEL = (5,5)
BOX_DILATE_ITER = 2

# parameters for lin extraction
LINE_DILATE_KERNEL = (1,20)
LINE_DILATE_ITER = 3

# model setting
VGG_MODEL = './textract/model/vgg/model.ckpt-19999'
VGG_DATA_DICT = './textract/model/vgg16.npy'
