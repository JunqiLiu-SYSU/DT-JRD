import os
import sys
import math
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
import csv

class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()

    def forward(self, preds, soft_labels, reduction='average'):
        """
        GDSL-based soft cross entropy loss  H(p,q)=-sum(p(x)log(q(x))), 
        where p(x) is the Gaussian distribution based soft labels, q(x) is the predicted possibility
        """
        batch = preds.shape[0]
        log_likelihood = -F.log_softmax(preds, dim=1)

        if reduction == 'average':
            loss = torch.sum(torch.mul(log_likelihood, soft_labels)) / batch
        else:
            loss = torch.sum(torch.mul(log_likelihood, soft_labels))
        return loss

def train_one_epoch(model, optimizer, data_loader, device, epoch, csv_filename, tb_writer):
    model.train()
    softceloss_function = SoftCrossEntropy()
    accu_loss = torch.zeros(1).to(device)
    MAE = torch.zeros(1).to(device)
    acc = torch.zeros(1).to(device)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, names, JRD_labels = data
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        correct_num = torch.sum(torch.eq(pred_classes, JRD_labels.to(device))).item()
        acc += correct_num/len(pred_classes)
        MAE += torch.abs(pred_classes - JRD_labels.to(device)).type(torch.FloatTensor).mean()

        # Gaussian distribution based soft labels
        x = torch.arange(0, 64).unsqueeze(0).repeat(images.shape[0], 1)  # [batchsize,64]
        sigma = 3  # you can alter the value of sigma
        u = JRD_labels.clone().detach().reshape([images.shape[0], 1])  # set the mean as JRD   [batchsize,1]
        softlabel = np.exp(-1 * ((x - u) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma)   # [batchsize,64]
        # Compute the sum of each batch, then normalize
        sums = torch.sum(softlabel, dim=1, keepdim=True)
        normalized_softlabel = softlabel / sums

        loss = softceloss_function(pred, normalized_softlabel.to(device))
        loss.backward()
        accu_loss += loss.detach()
        data_loader.desc = "[train{}]loss:{:.3f},MAE:{:.3f},acc:{:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            MAE.item() / (step + 1),
            acc.item() / (step + 1))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

        # update TensorBoard
        tags = ["train_loss", "train_MAE", "val_loss", "val_MAE", "learning_rate"]
        train_loss = accu_loss.item() / (step + 1)
        train_MAE = MAE.item() / (step + 1)
        tb_writer.add_scalar(tags[0], train_loss, epoch * 727 + step)
        tb_writer.add_scalar(tags[1], train_MAE, epoch * 727 + step)
        
        # save loss & MAE values to csv
        filename = csv_filename
        # Check whether the file exists to determine if the header needs to be written
        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='') as csvfile:
            fieldnames = ['epoch', 'step', 'train_loss', 'train_MAE']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()  # if the file does not exist, write the header
            writer.writerow({'epoch': epoch, 'step': step, 'train_loss': train_loss, 'train_MAE': train_MAE})

    return accu_loss.item() / (step + 1), MAE.item() / (step + 1), tb_writer

@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    softceloss_function = SoftCrossEntropy()
    model.eval()
    accu_loss = torch.zeros(1).to(device)
    MAE = torch.zeros(1).to(device)
    acc = torch.zeros(1).to(device)
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, names, JRD_labels = data 
        pred = model(images.to(device))           
        pred_classes = torch.max(pred, dim=1)[1]

        correct_num = torch.sum(torch.eq(pred_classes, JRD_labels.to(device))).item()
        acc += correct_num/len(pred_classes)
        MAE += torch.abs(pred_classes - JRD_labels.to(device)).type(torch.FloatTensor).mean()

        x = torch.arange(0, 64).unsqueeze(0).repeat(pred.shape[0], 1)
        sigma = 3
        u = JRD_labels.clone().detach().reshape([pred.shape[0], 1])
        softlabel = np.exp(-1 * ((x - u) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma)
        sums = torch.sum(softlabel, dim=1, keepdim=True)
        normalized_softlabel = softlabel / sums

        loss = softceloss_function(pred, normalized_softlabel.to(device))
        accu_loss += loss
        data_loader.desc = "[val{}]loss:{:.3f},MAE:{:.3f},acc:{:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            MAE.item() / (step + 1),
            acc.item() / (step + 1))

    return accu_loss.item() / (step + 1), MAE.item() / (step + 1)

@torch.no_grad()
def test(model, data_loader, device):
    softceloss_function = SoftCrossEntropy()
    model.eval()
    accu_loss = torch.zeros(1).to(device)
    MAE = torch.zeros(1).to(device)
    acc = torch.zeros(1).to(device)
    Pred_classes = []
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, names, JRD_labels = data
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        correct_num = torch.sum(torch.eq(pred_classes, JRD_labels.to(device))).item()
        acc += correct_num/len(pred_classes)
        MAE += torch.abs(pred_classes - JRD_labels.to(device)).type(torch.FloatTensor).mean()

        x = torch.arange(0, 64).unsqueeze(0).repeat(pred.shape[0], 1)
        sigma = 3 
        u = JRD_labels.clone().detach().reshape([pred.shape[0], 1])
        softlabel = np.exp(-1 * ((x - u) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma)
        sums = torch.sum(softlabel, dim=1, keepdim=True)
        normalized_softlabel = softlabel / sums
        
        loss = softceloss_function(pred, normalized_softlabel.to(device))
        accu_loss += loss
        Pred_classes.append(list(map(int, list(pred_classes.cpu().numpy()))))
        data_loader.desc = "loss: {:.3f}, MAE: {:.3f}, acc:{:.3f}".format(
            accu_loss.item() / (step + 1),
            MAE.item() / (step + 1),
            acc.item() / (step + 1))

    return accu_loss.item() / (step + 1), MAE.item() / (step + 1), Pred_classes