import os
import math
import argparse
import json
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from collections import OrderedDict
import numpy as np
import random

from model import CreateModel
from my_dataset import MyDataSet_d
from my_utils import train_one_epoch, evaluate

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def weights2model(k):
    k = int(k)
    if k <= 1:
       return ['0', str(k % 2)]
    elif k >= 2 and k <=5:
       return ['1', str((k-2) % 4)]
    elif k >= 6 and k <=9:
       return ['2', str((k-6) % 4)]
    elif k >= 10 and k <=15:
       return ['3', str((k-10) % 6)]

def main(args):
    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    # 解析 gpus 参数为整数列表
    args.gpus = [int(gpu.strip()) for gpu in args.gpus.split(',')]
    model = CreateModel(backbone=args.backbone, num_classes=args.num_classes, dropout_rate=args.dropout_rate, model_layers=args.model_layers)
    model = torch.nn.DataParallel(model, device_ids=args.gpus, output_device=args.gpus[0])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.weights != "":
        if args.backbone == 'Eff':
            pre_weights = os.path.join(args.weights, 'pre_efficientnetv2-s.pth')
        elif args.backbone == 'Res':
            pre_weights = os.path.join(args.weights, 'resnet' + str(args.model_layers) + '.pth')
        if os.path.exists(pre_weights):
            weights_dict = torch.load(pre_weights, map_location=device)
            model_dict = dict(model.state_dict().items())
            model_keys = list(model_dict.keys())
            preweights_dict = dict(weights_dict.items())
            new_load_weights_dict = OrderedDict()
            if args.backbone == 'Eff':
                load_steam_weights_dict = {k: v for k, v in preweights_dict.items() if 'head' not in k and 'blocks' not in k and 'module.EB.' + k in model_keys}
                load_steam_weights_dict = {k: v for k, v in load_steam_weights_dict.items() if model.state_dict()['module.EB.' + k].numel() == v.numel()}
                load_block_weights_dict = {k: v for k, v in preweights_dict.items() if 'blocks' in k and int(k.split('.')[1]) < 16 and 'module.EB.blocks' + weights2model(k.split('.')[1])[0] + '.' + weights2model(k.split('.')[1])[1] + k[7+len(k.split('.')[1]):] in model_keys}
                load_block_weights_dict = {k: v for k, v in load_block_weights_dict.items() if
                                           model.state_dict()['module.EB.blocks' + weights2model(k.split('.')[1])[0] + '.' + weights2model(k.split('.')[1])[1] + k[7+len(k.split('.')[1]):]].numel() == v.numel()}
                for k, v in load_steam_weights_dict.items():
                    name = 'module.EB.' + k
                    new_load_weights_dict[name] = v
                for k, v in load_block_weights_dict.items():
                    name = 'module.EB.blocks' + weights2model(k.split('.')[1])[0] + '.' + weights2model(k.split('.')[1])[1] + k[7+len(k.split('.')[1]):]  # 添加module.EB.
                    new_load_weights_dict[name] = v
            elif args.backbone == 'Res':
                load_weights_dict = {k: v for k, v in preweights_dict.items() if
                                     'head' not in k and 'module.EB.' + k in model_keys}
                load_weights_dict = {k: v for k, v in load_weights_dict.items() if
                                     model.state_dict()['module.EB.' + k].numel() == v.numel()}
                for k, v in load_weights_dict.items():
                    name = 'module.EB.' + k
                    new_load_weights_dict[name] = v
            print(model.load_state_dict(new_load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(pre_weights))
    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "EB" in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    tb_writer = SummaryWriter()
    # if os.path.exists("./weights") is False:
    #     os.makedirs("./weights")

    with open(os.path.join(args.datainfo_path, 'train.json'), 'r') as f:
        distorted_train_dist = json.load(f)
    with open(os.path.join(args.datainfo_path, 'val.json'), 'r') as f:
        distorted_val_dist = json.load(f)

    size = 640
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.CenterCrop((size, size))]),
        "val": transforms.Compose([transforms.ToTensor(),
                                   transforms.CenterCrop((size, size))]),
    }
    normalization = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    distorted_train_paths = distorted_train_dist['paths']
    distorted_train_labels = distorted_train_dist['labels']
    distorted_train_names = distorted_train_dist['names']

    distorted_val_paths = distorted_val_dist['paths']
    distorted_val_labels = distorted_val_dist['labels']
    distorted_val_names = distorted_val_dist['names']

    distorted_train_dataset = MyDataSet_d(images_paths=distorted_train_paths,
                                          images_labels=distorted_train_labels,
                                          images_names=distorted_train_names,
                                          transform=data_transform["train"],
                                          normalization=normalization)

    distorted_val_dataset = MyDataSet_d(images_paths=distorted_val_paths,
                                        images_labels=distorted_val_labels,
                                        images_names=distorted_val_names,
                                        transform=data_transform["val"],
                                        normalization=normalization)

    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    nw = 32
    #nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 10])

    distorted_train_loader = torch.utils.data.DataLoader(distorted_train_dataset,
                                                         batch_size=train_batch_size,
                                                         shuffle=True,
                                                         pin_memory=True,
                                                         num_workers=nw,
                                                         collate_fn=distorted_train_dataset.collate_fn)

    distorted_val_loader = torch.utils.data.DataLoader(distorted_val_dataset,
                                                       batch_size=val_batch_size,
                                                       shuffle=False,
                                                       pin_memory=True,
                                                       num_workers=nw,
                                                       collate_fn=distorted_val_dataset.collate_fn)
    
    tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
    
    for epoch in range(args.epochs):
         # train
         train_loss, train_acc = train_one_epoch(model=model,
                                                 optimizer=optimizer,
                                                 data_path=args.data_path,
                                                 distorted_data_loader=distorted_train_loader,
                                                 device=device,
                                                 epoch=epoch,
                                                 size=size)
         scheduler.step()

         # val
         val_loss, val_acc = evaluate(model=model,
                                     distorted_data_loader=distorted_val_loader,
                                     data_path=args.data_path,
                                     device=device,
                                     epoch=epoch,
                                     size=size)

         
         tb_writer.add_scalar(tags[0], train_loss, epoch)
         tb_writer.add_scalar(tags[1], train_acc, epoch)
         tb_writer.add_scalar(tags[2], val_loss, epoch)
         tb_writer.add_scalar(tags[3], val_acc, epoch)
         tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
         if args.backbone == 'Eff':
             torch.save(model.state_dict(),
                        args.save_model_path + '/' + args.backbone + '/Eff-{}.pth'.format(epoch))
         elif args.backbone == 'Res':
             torch.save(model.state_dict(),
                        args.save_model_path + '/' + args.backbone + '/Res' + str(args.model_layers) + '-{}.pth'.format(epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', type=str, default='cuda:2', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--gpus', type=str, default='2')
    parser.add_argument('--backbone', type=str, default='Eff', help='Eff, Res')
    parser.add_argument('--model_layers', type=int, default=34, help='18, 34, 50')
    parser.add_argument('--datainfo_path', type=str, default='./jsonfiles')
    parser.add_argument('--save_model_path', type=str, default='./train_weights')
    parser.add_argument('--data-path', type=str, default="./data")
    parser.add_argument('--weights', type=str, default='./pre_weights',
                        help='initial weights path')
    parser.add_argument('--seed', type=int, default=42)
    opt = parser.parse_args()
    setup_seed(opt.seed)
    main(opt)
