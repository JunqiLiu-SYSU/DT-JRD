import os
import cv2
import json
import copy
import torch
import argparse
import numpy as np
from torchvision import transforms
from dataset import MyDataSet
from model import vit_large_patch32_224_in21k as create_model
from utils import test
import pandas as pd

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class PJRD():
    def __init__(self, opt):
        self.test_dist = []
        self.GT_JRDs_dist = []
        self.objectinfo = []
        self.images_paths = []
        self.images_names = []
        self.model = []

    def prepare_data(self):
        with open('./jsonfiles/test_names.json', 'r') as f:
            self.test_dist = json.load(f)
        with open('./jsonfiles/JRD_info.json', 'r') as f:
            self.GT_JRDs_dist = json.load(f)
        with open('./jsonfiles/all_objects_infos.json', 'r') as f:
            self.objectinfo = json.load(f)

        self.test_names = self.test_dist['names']
        self.test_paths = []
        for k in range(len(self.test_names)):
            test_path = opt.data_path + self.test_names[k] + '/' + self.test_names[k] + '.png'
            self.test_paths.append(test_path)

    def creat_model(self):

        opt.gpus = [int(gpu.strip()) for gpu in opt.gpus.split(',')]
        self.model = create_model(num_classes=64, img_size=opt.size, patch_size=opt.patch_size, blocks=opt.block, has_logits=False)
        self.model = torch.nn.DataParallel(self.model, device_ids=opt.gpus, output_device=opt.gpus[0])
        self.model.to(opt.device)
            
        if opt.train_weights != "":
            if os.path.exists(opt.train_weights):
                weights_dict = torch.load(opt.train_weights, map_location=opt.device)
                load_weights_dict = {k: v for k, v in weights_dict.items()
                                     if self.model.state_dict()[k].numel() == v.numel()}
                print(self.model.load_state_dict(load_weights_dict, strict=False))
            else:
                raise FileNotFoundError("not found weights file: {}".format(opt.train_weights))        

    def predict(self):
        size = opt.size
        nw = 16
        data_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((size, size), antialias=True),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_paths = self.test_paths
        test_names = self.test_names

        test_dataset = MyDataSet(images_paths=test_paths,
                                    images_names=test_names,
                                    JRD_info_dict=self.GT_JRDs_dist,
                                    transform=data_transform)

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=opt.batch_size,
                                                    shuffle=False,
                                                    pin_memory=False,
                                                    num_workers=nw,
                                                    collate_fn=test_dataset.collate_fn)

        test_loss, test_acc, Pred_classes = test(model=self.model,
                                                 data_loader=test_loader,
                                                 device=opt.device)
        return test_loss, test_acc, Pred_classes

    def analyses(self):
        self.prepare_data()
        self.creat_model()
        test_loss, test_acc, Pred_classes = self.predict()

        with open('./jsonfiles/all_GT_classes.json', 'r') as f:
            GT_classes = json.load(f)

        new_Pred_classes = []
        for Pred_class in Pred_classes:
            new_Pred_classes += Pred_class

        GT_classes = GT_classes['gt']
        new_GT_classes = []
        for GT_class in GT_classes:
            new_GT_classes += GT_class

        GT_classes = new_GT_classes        
        Pred_classes = new_Pred_classes    
        abs_delta_JRDs = np.abs(np.array(Pred_classes) - np.array(GT_classes)) 
        print('E_A:', "{:.3f}".format(np.mean(abs_delta_JRDs)))

        # calculate MAE for whose ground truth JRD belongs to [27,51]
        index_not_in_2751 = [i for i in range(len(GT_classes)) if GT_classes[i] < 27 or GT_classes[i] > 51]
        new_pre_JRDs = [Pred_classes[i] for i in range(len(Pred_classes)) if i not in index_not_in_2751]
        new_gt_JRDs = [GT_classes[i] for i in range(len(GT_classes)) if i not in index_not_in_2751]
        new_pre_JRDs = np.array(new_pre_JRDs)
        new_gt_JRDs = np.array(new_gt_JRDs)
        part_pre_MAE2751 = np.mean(abs(new_gt_JRDs - new_pre_JRDs))
        QP_abs_dist = {k: [] for k in list(range(64))}
        for i, GT_JRD in enumerate(GT_classes):
            QP_abs_dist[GT_JRD].append(abs_delta_JRDs[i])
        QP_err_mean = [round(np.mean(v), 2) for k, v in QP_abs_dist.items()]
        QP_err_sum = [np.sum(v) for k, v in QP_abs_dist.items()]
        QP_err_num = [len(v) for k, v in QP_abs_dist.items()]
        print('E[27,51]:', "{:.3f}".format(part_pre_MAE2751))

        objectclasses = []
        objectsizes = []
        for test_target_name in self.test_names:
            objectclass = self.objectinfo[test_target_name]['class']
            objectclasses.append(objectclass)              
            test_object_path = os.path.join(opt.data_path, test_target_name, test_target_name + '.png')
            test_object_img = cv2.imread(test_object_path)
            height = test_object_img.shape[0]
            width = test_object_img.shape[1]
            objectsizes.append(height * width)            

        index_smallobject = [i for i in range(len(objectsizes)) if objectsizes[i] < 32 * 32]
        index_medobject = [i for i in range(len(objectsizes)) if objectsizes[i] >= 32 * 32 and objectsizes[i] < 96 * 96]
        index_largeobject = [i for i in range(len(objectsizes)) if objectsizes[i] >= 96 * 96]

        print('small:', len(index_smallobject),
              "{:.3f}".format(np.mean([abs_delta_JRDs[i] for i in range(len(abs_delta_JRDs)) if i in index_smallobject])))
        print('medium:', len(index_medobject),
              "{:.3f}".format(np.mean([abs_delta_JRDs[i] for i in range(len(abs_delta_JRDs)) if i in index_medobject])))
        print('large:', len(index_largeobject),
              "{:.3f}".format(np.mean([abs_delta_JRDs[i] for i in range(len(abs_delta_JRDs)) if i in index_largeobject])))

        index_peopleobject = [i for i in range(len(objectclasses)) if objectclasses[i] == 0]
        index_carobject = [i for i in range(len(objectclasses)) if objectclasses[i] == 2]
        print('people:', len(index_peopleobject),
              "{:.3f}".format(
                  np.mean([abs_delta_JRDs[i] for i in range(len(abs_delta_JRDs)) if i in index_peopleobject])))
        print('car:', len(index_carobject),
              "{:.3f}".format(np.mean([abs_delta_JRDs[i] for i in range(len(abs_delta_JRDs)) if i in index_carobject])))

        with open('./jsonfiles/coco80_indices.json', 'r') as f:
            class_map = json.load(f)
        class_name = []
        class_num = []
        mean_abs_delta = []
        for j in range(80):
            abs_delta = [abs_delta_JRDs[i] for i in range(len(abs_delta_JRDs)) if objectclasses[i] == j]
            # print(class_map[str(j)], ':', len(abs_delta), "{:.3f}".format(np.mean(abs_delta)))
            class_name.append(class_map[str(j)])
            class_num.append(len(abs_delta))
            mean_abs_delta.append("{:.3f}".format(np.mean(abs_delta)))

        # save results to Excel
        # create DataFrame
        df1 = pd.DataFrame({
            'GT_classes': GT_classes,
            'Pred_classes': Pred_classes,
            'delta': np.array(Pred_classes) - np.array(GT_classes),
            'abs_delta_JRDs': abs_delta_JRDs
        })
        df2 = pd.DataFrame({
            'QP': [i for i in range(64)],
            'QP_err_mean': QP_err_mean,
            'QP_num': QP_err_num
        })
        df3 = pd.DataFrame({
            'class_name': class_name,
            'class_num': class_num,
            'mean_abs_delta': mean_abs_delta
        })
        
        if not os.path.exists(opt.save_path):
            os.makedirs(opt.save_path)
        with pd.ExcelWriter(os.path.join(opt.save_path, 'results.xlsx'), engine='openpyxl') as writer:
            df1.to_excel(writer, sheet_name='Sheet1', index=False)
            df2.to_excel(writer, sheet_name='Sheet2', index=False)
            df3.to_excel(writer, sheet_name='Sheet3', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--train_weights', type=str,
                        default='./train_weights/ViT-L32/block24-patch_size32-lr0.01-bs32-MAE5.3169.pth')
    parser.add_argument('--data_path', type=str, default='./data/original/')
    parser.add_argument('--save_path', type=str, default='./predicted_results')
    parser.add_argument('--block', type=int, default=24)
    parser.add_argument('--size', type=int, default=384)
    opt = parser.parse_args()
    pjrd = PJRD(opt)
    pjrd.analyses()
