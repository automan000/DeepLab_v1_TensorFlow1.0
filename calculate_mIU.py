import numpy as np
from scipy import ndimage

N_CLASSES_PASCAL = 21
DATA_LIST='./images_val/'

"""
def computeIoU(y_pred_batch, y_true_batch):
    return np.mean(np.asarray([pixelAccuracy(y_pred_batch[i], y_true_batch[i]) for i in range(len(y_true_batch))]))

def pixelAccuracy(y_pred, y_true):
    y_pred = np.argmax(np.reshape(y_pred,[N_CLASSES_PASCAL,img_rows,img_cols]),axis=0)
    y_true = np.argmax(np.reshape(y_true,[N_CLASSES_PASCAL,img_rows,img_cols]),axis=0)
    y_pred = y_pred * (y_true>0)

    return 1.0 * np.sum((y_pred==y_true)*(y_true>0)) /  np.sum(y_true>0)
"""
def main():
    groundtruth = []
    predictions = []

    import glob

    gt_filename = glob.glob(DATA_LIST+'mask/*.png')
    pd_filename = glob.glob(DATA_LIST+'pred/*.png')

    for filename in gt_filename:  # assuming gif
        corresponding_pred = DATA_LIST + 'pred/' + filename[filename.rfind('/') + 1:]
        if corresponding_pred in pd_filename:
            im = ndimage.imread(filename)
            groundtruth.append(im)
            im = ndimage.imread(corresponding_pred)
            predictions.append(im)
    print('test')


if __name__ == '__main__':
    main()
