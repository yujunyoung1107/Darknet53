import argparse
import sys, os
import time
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data.dataloader import DataLoader
import os; os.environ['WANDB_NOTEBOOK_NAME'] = 'some text here'

from model.models import *
from loss.loss import *
from utils.utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="Darknet53")
    parser.add_argument('--mode', dest='mode', help="train / eval",
                        default=None, type=str)
    parser.add_argument('--pretrained_freeze', dest='pretrained_freeze', help="freezing pretrained parameter",
                        default=True, type=bool)             
    parser.add_argument('--data', dest='data', help="data directory",
                        default='./dogcat', type=str)
    parser.add_argument('--output_dir', dest='output_dir', help="output directory",
                        default='./output', type=str)
    parser.add_argument('--checkpoint', dest='checkpoint', help="checkpoint trained model",
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    args = parser.parse_args()
    return args


def main():
    print(torch.__version__)

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    if torch.cuda.is_available():
        print("gpu")
        device = torch.device('cuda')
    else:
        print("cpu")
        device = torch.device('cpu')

    train_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    valid_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    train_dir = os.path.join(args.data, 'train')
    valid_dir = os.path.join(args.data, 'eval')

    train_dataset = datasets.ImageFolder(train_dir, train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, valid_transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=32,
                              num_workers=4,
                              pin_memory=True,
                              drop_last=True,
                              shuffle=True)
    eval_loader = DataLoader(valid_dataset,
                             batch_size=1,
                             num_workers=0,
                             pin_memory=True,
                             drop_last=False,
                             shuffle=True)

    Darknet53 = get_model('Darknet53')
    model = Darknet53(2)

    config = dict()
    config = {"pretrain layer freeze" : args.pretrained_freeze}


    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        pretrained_state_dict = checkpoint['state_dict']
        model_state_dict = model.state_dict()
        for key, value in pretrained_state_dict.items():
            if key == 'fc.weight' or key == 'fc.bias':
                continue
            else:
                model_state_dict[key] = value
        
        model.load_state_dict(model_state_dict)

        if args.pretrained_freeze:    
            for param in model.parameters():
                param.requires_grad = False

            for param in model.fc.parameters():
                param.requires_grad = True

        
    if args.mode == 'train':

        wandb.init(project = 'Darknet53 training', config = config)

        # loss와 accuracy를 계산해주는 class
        calc = calc_accuracy_loss()
        test_calc = calc_accuracy_loss()

        model.to(device)
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = get_criterion(crit="mnist", device=device)

        epoch = 1
        log_dict = {}

        start_time = time.time()
        for e in range(epoch):
            for i, batch in enumerate(train_loader):
                img = batch[0]
                gt = batch[1]

                img = img.to(device)
                gt = gt.to(device)

                out = model(img)

                calc.update(gt, out)

                loss_val = criterion(out, gt)

                loss_val.backward()
                optimizer.step()
                optimizer.zero_grad()

                calc.update_loss(loss_val.item())
                print(i)

                if len(calc) == 5:
                    log_dict = {}
                    accuracy, loss = calc.return_acc_loss()
                    calc.reset()
                    print("{} epoch / {}th iteration Loss : {} | Accuracy : {}".format(e, i, loss, accuracy))

                    log_dict["train loss"] = loss
                    log_dict["train accuracy"] = accuracy
                    wandb.log(log_dict)

                if i % 10 == 0:
                    model.eval()
                    log_dict = {}
                    with torch.no_grad():
                        for i, batch in enumerate(eval_loader):
                            img = batch[0]
                            gt = batch[1]

                            img = img.to(device)
                            gt = gt.to(device)

                            out = model(img)
                            test_calc.update(gt, out)

                            loss_val = criterion(out, gt)
                            test_calc.update_loss(loss_val.item())

                    accuracy, loss = test_calc.return_acc_loss()
                    test_calc.reset()
                    print("test loss : {} accuracy : {}".format(loss, accuracy))

                    log_dict["test loss"] = loss
                    log_dict["test accuracy"] = accuracy
                    wandb.log(log_dict)
                    model.train()


        print('training time : {}'.format(time.time() - start_time))

if __name__ == "__main__":
    args = parse_args()
    main()