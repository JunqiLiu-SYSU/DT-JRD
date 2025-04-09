"""
Training codes for DT-JRD model
"""
import os
import math
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from dataset import MyDataSet
from model import vit_large_patch32_224_in21k as create_model
from utils import train_one_epoch, evaluate
from collections import OrderedDict
import numpy as np
import random
import datetime

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Invalid value for a boolean argument. Please use true/false, yes/no, t/f, y/n, 1/0.')

def interpolate_embeddings(
    image_size: int,
    patch_size: int,
    model_state: "OrderedDict[str, torch.Tensor]",
    interpolation_mode: str = "bicubic", #"linear"
    reset_heads: bool = False,
) -> "OrderedDict[str, torch.Tensor]":
    """This function helps interpolate positional embeddings during checkpoint loading,
    especially when you want to apply a pre-trained model on images with different resolution.

    Args:
        image_size (int): Image size of the new model.
        patch_size (int): Patch size of the new model.
        model_state (OrderedDict[str, torch.Tensor]): State dict of the pre-trained model.
        interpolation_mode (str): The algorithm used for upsampling. Default: bicubic.
        reset_heads (bool): If true, not copying the state of heads. Default: False.

    Returns:
        OrderedDict[str, torch.Tensor]: A state dict which can be loaded into the new model.
    """
    # Shape of pos_embedding is (1, seq_length, hidden_dim)
    pos_embedding = model_state["module.pos_embed"]
    n, seq_length, hidden_dim = pos_embedding.shape
    if n != 1:
        raise ValueError(f"Unexpected position embedding shape: {pos_embedding.shape}")

    new_seq_length = (image_size // patch_size) ** 2 + 1

    # Need to interpolate the weights for the position embedding.
    # We do this by reshaping the positions embeddings to a 2d grid, performing
    # an interpolation in the (h, w) space and then reshaping back to a 1d grid.
    if new_seq_length != seq_length:
        # The class token embedding shouldn't be interpolated, so we split it up.
        seq_length -= 1
        new_seq_length -= 1
        pos_embedding_token = pos_embedding[:, :1, :]
        pos_embedding_img = pos_embedding[:, 1:, :]

        # (1, seq_length, hidden_dim) -> (1, hidden_dim, seq_length)
        pos_embedding_img = pos_embedding_img.permute(0, 2, 1)
        seq_length_1d = int(math.sqrt(seq_length))
        if seq_length_1d * seq_length_1d != seq_length:
            raise ValueError(
                f"seq_length is not a perfect square! Instead got seq_length_1d * seq_length_1d = {seq_length_1d * seq_length_1d } and seq_length = {seq_length}"
            )

        # (1, hidden_dim, seq_length) -> (1, hidden_dim, seq_l_1d, seq_l_1d)
        pos_embedding_img = pos_embedding_img.reshape(1, hidden_dim, seq_length_1d, seq_length_1d)
        new_seq_length_1d = image_size // patch_size

        # Perform interpolation.
        # (1, hidden_dim, seq_l_1d, seq_l_1d) -> (1, hidden_dim, new_seq_l_1d, new_seq_l_1d)
        new_pos_embedding_img = nn.functional.interpolate(
            pos_embedding_img,
            size=new_seq_length_1d,
            mode=interpolation_mode,
            align_corners=True,
        )

        # (1, hidden_dim, new_seq_l_1d, new_seq_l_1d) -> (1, hidden_dim, new_seq_length)
        new_pos_embedding_img = new_pos_embedding_img.reshape(1, hidden_dim, new_seq_length)

        # (1, hidden_dim, new_seq_length) -> (1, new_seq_length, hidden_dim)
        new_pos_embedding_img = new_pos_embedding_img.permute(0, 2, 1)

        new_pos_embedding = torch.cat([pos_embedding_token, new_pos_embedding_img], dim=1)
        model_state["module.pos_embed"] = new_pos_embedding

        if reset_heads:
            model_state_copy: "OrderedDict[str, torch.Tensor]" = OrderedDict()
            for k, v in model_state.items():
                if not k.startswith("heads"):
                    model_state_copy[k] = v
            model_state = model_state_copy

    return model_state

def main(args):

    tb_writer = SummaryWriter()
    size = args.size
    data_transform = {
        "train": transforms.Compose([transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Resize((size, size), antialias=True),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize((size, size), antialias=True),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    with open(os.path.join(args.datainfo_path, 'train_names.json'), 'r') as f:
        train_dist = json.load(f)
    with open(os.path.join(args.datainfo_path, 'val_names.json'), 'r') as f:
        val_dist = json.load(f)
    with open(os.path.join(args.datainfo_path, 'JRD_info.json'), 'r') as f:
        JRD_dict = json.load(f)

    train_names = train_dist["names"]
    # Note: Please change to the path where the dataset is stored
    train_paths = []
    for k in range(len(train_names)):
        train_path = args.data_path + train_names[k] + '/' + train_names[k] + '.png'
        train_paths.append(train_path)

    val_names = val_dist["names"]
    val_paths = []
    for k in range(len(val_names)):
        val_path = args.data_path + val_names[k] + '/' + val_names[k] + '.png'
        val_paths.append(val_path)

    # Instantiating the training set
    train_dataset = MyDataSet(images_paths=train_paths,
                                          images_names=train_names,
                                          JRD_info_dict=JRD_dict,
                                          transform=data_transform["train"])
    # Instantiating the validation set
    val_dataset = MyDataSet(images_paths=val_paths,
                                        images_names=val_names,
                                        JRD_info_dict=JRD_dict,
                                        transform=data_transform["val"])
    batch_size = args.batch_size
    nw = 16     # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                pin_memory=True,
                                                num_workers=nw,
                                                collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                pin_memory=True,
                                                num_workers=nw,
                                                collate_fn=val_dataset.collate_fn)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.gpus = [int(gpu.strip()) for gpu in args.gpus.split(',')]  # Parsing gpus arguments as a list of integers
    model = create_model(num_classes=64, img_size=args.size, patch_size=args.patch_size, blocks=args.block, has_logits=False,
                         drop_ratio=args.drop_ratio, attn_drop_ratio=args.attn_drop_ratio, drop_path_ratio=args.drop_path_ratio)
    model = torch.nn.DataParallel(model, device_ids=args.gpus, output_device=args.gpus[0])
    model = model.to(device)

    # 多卡
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        parallel_weights_dict = OrderedDict()
        for k,v in weights_dict.items():
            parallel_weights_dict['module.' + k] = v
        for k in ['module.pre_logits.fc.weight', 'module.pre_logits.fc.bias', 'module.head.weight', 'module.head.bias']:
            del parallel_weights_dict[k]

        # Performing position embedding interpolation if the image size is not 224
        if args.size != 224:
            parallel_weights_dict = interpolate_embeddings(image_size=args.size, patch_size=args.patch_size, model_state=parallel_weights_dict)
        print(model.load_state_dict(parallel_weights_dict, strict=False))

    if str2bool(args.freeze_layers):
        for name, para in model.named_parameters():
            if "blocks" in name or 'head' in name:
                print("training {}".format(name))
            else:
                para.requires_grad_(False)
                print("not training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine learning rate decay
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_MAE = 100  # Setting the initial value of best_MAE
    best_loss = 100 # Setting the initial value of best_loss
    patience = 3  # Early stopping patience
    counter = 0  # Counter for early stopping
    tags = ["train_loss", "train_MAE", "val_loss", "val_MAE", "learning_rate"]
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('Training start at',time) # record the time
    for epoch in range(args.epochs):
        # train
        print('epoch:',epoch,'Current learning rate:', optimizer.param_groups[0]['lr'])
        train_loss, train_MAE, tb_writer = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                csv_filename=args.csv_filename,
                                                tb_writer=tb_writer)
        scheduler.step()

        # validate
        val_loss, val_MAE = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        
        # Update TensorBoard
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_MAE, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        print('epoch:',epoch,'train_loss:',train_loss,'train_MAE:',train_MAE,'val_loss:',val_loss,'val_MAE:',val_MAE)

        # Save the model only when the MAE of the validation set decreases and update the counter for early stopping.
        if val_MAE < best_MAE or val_loss < best_loss:
            if val_MAE < best_MAE:
                best_MAE = val_MAE      # update best_MAE
                best_epoch = epoch      # update best_epoch
            if val_loss < best_loss:
                best_loss = val_loss    # update best_loss
            counter = 0
            # Construct save path
            save_dir = f"{args.save_model_path}/{args.backbone}"
            save_path = f"{save_dir}/block{args.block}-patch_size{args.patch_size}-lr{args.lr}-bs{args.batch_size}-MAE{val_MAE:.4f}.pth"
            # Check if the folder exists, if not, create it
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # save model's checkpoint
            torch.save(model.state_dict(), save_path)  
        else:
            counter += 1
        # Check if the condition for early stopping is met
        if counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break
    print('best_epoch=',best_epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--datainfo_path', type=str, default='./jsonfiles')
    parser.add_argument('--save_model_path', type=str, default='./train_weights')
    parser.add_argument('--data_path', type=str,
                        default="./data/original/")
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--weights', type=str, default='./pre_weights/jx_vit_large_patch32_224_in21k-9046d2e7.pth',
                        help='initial weights path')
    parser.add_argument('--freeze_layers', type=str, default=True)
    parser.add_argument('--gpus', type=str, default='0,1')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--backbone', type=str, default='ViT-L32')
    parser.add_argument('--block', type=int, default=24)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--size', type=int, default=384)
    parser.add_argument('--drop_ratio', type=float, default=0.)
    parser.add_argument('--attn_drop_ratio', type=float, default=0.)
    parser.add_argument('--drop_path_ratio', type=float, default=0.)
    parser.add_argument('--csv_filename', default='training_metrics.csv')
    opt = parser.parse_args()
    setup_seed(opt.seed)
    main(opt)