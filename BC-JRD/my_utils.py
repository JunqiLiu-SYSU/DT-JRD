import os
import sys
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

def train_one_epoch(model, optimizer, data_path, distorted_data_loader, device, epoch, size):
    model.train()
    celoss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.CenterCrop((size, size)),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    sample_num = 0
    original_data_path = os.path.join(data_path, 'original')
    distorted_data_loader = tqdm(distorted_data_loader)
    for distorted_step, distorted_data in enumerate(distorted_data_loader):
        distorted_images, distorted_labels, distorted_names = distorted_data
        original_imgs = []
        for kk, distorted_name in enumerate(distorted_names):
            original_img_path = os.path.join(original_data_path, distorted_name, distorted_name + '.png')
            original_img = Image.open(original_img_path)
            if original_img.mode != 'RGB':
                original_img.convert('RGB')
            original_img = data_transform(original_img)
            original_imgs.append(original_img)
        original_images = torch.stack(original_imgs, dim=0)
        pred = model(original_images.to(device), distorted_images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, distorted_labels.to(device)).sum()
        sample_num += distorted_images.shape[0]
        loss = celoss_function(pred, distorted_labels.to(device))
        loss.backward()
        accu_loss += loss.detach()
        distorted_data_loader.desc = "[train{}]loss:{:.3f},acc:{:.3f}".format(
            epoch,
            accu_loss.item() / (distorted_step + 1),
            accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (distorted_step + 1), accu_num.item() / sample_num

@torch.no_grad()
def evaluate(model, distorted_data_loader, data_path, device, epoch, size):
    celoss_function = torch.nn.CrossEntropyLoss()
    model.eval()
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.CenterCrop((size, size)),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    sample_num = 0
    distorted_data_loader = tqdm(distorted_data_loader)
    original_data_path = os.path.join(data_path, 'original')
    for distorted_step, distorted_data in enumerate(distorted_data_loader):
        distorted_images, distorted_labels, distorted_names = distorted_data
        original_imgs = []
        for distorted_name in distorted_names:
            original_img_path = os.path.join(original_data_path, distorted_name, distorted_name + '.png')
            original_img = Image.open(original_img_path)
            if original_img.mode != 'RGB':
                original_img.convert('RGB')
            original_img = data_transform(original_img)
            original_imgs.append(original_img)
        original_images = torch.stack(original_imgs, dim=0)
        pred = model(original_images.to(device), distorted_images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, distorted_labels.to(device)).sum()
        sample_num += distorted_images.shape[0]
        loss = celoss_function(pred, distorted_labels.to(device))
        accu_loss += loss
        distorted_data_loader.desc = "[val{}]loss:{:.3f},acc:{:.3f}".format(epoch,
                                                                            accu_loss.item() / (distorted_step + 1),
                                                                            accu_num.item() / sample_num)
    return accu_loss.item() / (distorted_step + 1), accu_num.item() / sample_num

@torch.no_grad()
def test(model, distorted_data_loader, data_path, device, size):
    celoss_function = torch.nn.CrossEntropyLoss()
    model.eval()
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.CenterCrop((size, size)),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    Pred_classes = []
    sample_num = 0
    distorted_data_loader = tqdm(distorted_data_loader)
    original_data_path = os.path.join(data_path, 'original')
    for distorted_step, distorted_data in enumerate(distorted_data_loader):
        distorted_images, distorted_labels, distorted_names = distorted_data
        original_imgs = []
        for distorted_name in distorted_names:
            original_img_path = os.path.join(original_data_path, distorted_name, distorted_name + '.png')
            original_img = Image.open(original_img_path)
            if original_img.mode != 'RGB':
                original_img.convert('RGB')
            original_img = data_transform(original_img)
            original_imgs.append(original_img)
        original_images = torch.stack(original_imgs, dim=0)
        pred = model(original_images.to(device), distorted_images.to(device))

        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, distorted_labels.to(device)).sum()
        sample_num += distorted_images.shape[0]

        loss = celoss_function(pred, distorted_labels.to(device))
        accu_loss += loss

        Pred_classes.append(list(map(int, list(pred_classes.cpu().numpy()))))
        distorted_data_loader.desc = "loss: {:.3f}, acc: {:.3f}".format(accu_loss.item() / (distorted_step + 1),
                                                                        accu_num.item() / sample_num)
    return accu_loss.item() / (distorted_step + 1), accu_num.item() / sample_num, Pred_classes




