import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image, ImageDraw

def show_img(img_list, gt_list, est_list):

    fig, ax = plt.subplots(1, 5)
    for idx, (img_data, est, gt) in enumerate(zip(img_list, gt_list, est_list)):
        _img_data = img_data * 255

        _img_data = np.array(_img_data[0,0], dtype=np.uint8)

        img_data = Image.fromarray(_img_data)

        ax[idx].set_title("Estimation : {}\n".format(est) +  "Answer : {}\n".format(gt))
        ax[idx].imshow(img_data)

    plt.show()

class calc_accuracy_loss:
    
    def __init__(self):
        self.num_data = 0
        self.num_correct = 0
        self.loss = 0.
        self.count = 0

    def reset(self):
        self.num_data = 0
        self.num_correct = 0
        self.loss = 0.
        self.count = 0

    def update(self, y, y_hat):
        self.num_data += len(y_hat)
        correct_num = (torch.argmax(y_hat, dim=1) == y).sum().item()
        self.num_correct += correct_num

    def update_loss(self, loss):
        self.loss += loss
        self.count += 1

    def return_acc_loss(self):
        return self.num_correct / self.num_data * 100, self.loss / self.count

    def __len__(self):
        return self.count