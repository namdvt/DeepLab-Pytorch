import matplotlib.pyplot as plt
import numpy as np


def write_figures(location, train_losses, val_losses):
    plt.plot(train_losses, label='training loss')
    plt.plot(val_losses, label='validation loss')
    plt.legend()
    plt.savefig(location + '/loss.png')
    plt.close('all')


def write_log(location, epoch, train_loss, val_loss):
    if epoch == 0:
        f = open(location + '/log.txt', 'w+')
        f.write('epoch\t\ttrain_loss\t\tval_loss\n')
    else:
        f = open(location + '/log.txt', 'a+')

    f.write(str(epoch) + '\t' + str(train_loss) + '\t' + str(val_loss) + '\n')

    f.close()


def mask_to_anns_img(binary_mask):
    _, height, width = binary_mask.shape
    anns_img = np.zeros((height, width))
    for i in range(90):
        anns_img = np.maximum(anns_img, binary_mask[i]*(i + 1))
    return anns_img