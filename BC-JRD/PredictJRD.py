import os
import cv2
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from natsort import natsorted
from torchvision import transforms
from model import CreateModel
from my_dataset import MyDataSet_d
from my_utils import test
import time

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
    def __init__(self, args):
        self.distorted_test_dist = []
        self.GT_JRDs_dist = []
        self.objectinfo = []
        self.images_paths = []
        self.images_names = []
        self.JRDs = []
        self.model = []
        self.args = args

    def prepare_data(self):
        with open(self.args.test_info_path, 'r') as f:
            self.distorted_test_dist = json.load(f)
        with open(self.args.GroundTrue_JRD_path, 'r') as f:
            self.GT_JRDs_dist = json.load(f)
        with open(self.args.objectinfo_json_path, 'r') as f:
            self.objectinfo = json.load(f)

        self.images_paths = self.distorted_test_dist['paths']
        self.images_names = self.distorted_test_dist['names']
        self.images_names = natsorted(list(set(self.images_names)))

        for key, value in self.GT_JRDs_dist.items():
            self.JRDs.append(value)

    def creat_model(self):
        self.model = CreateModel(backbone=self.args.backbone, num_classes=2, dropout_rate=self.args.dropout_rate, model_layers=self.args.model_layers)
        #self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu, output_device=self.args.gpu)
        self.model = torch.nn.DataParallel(self.model, device_ids=[self.args.gpu], output_device=[self.args.gpu])
        self.model.to(self.args.device)
        if self.args.weights != "":
            if os.path.exists(self.args.weights):
                weights_dict = torch.load(self.args.weights, map_location=self.args.device)
                load_weights_dict = {k: v for k, v in weights_dict.items() 
                                     if self.model.state_dict()[k].numel() == v.numel()}
                self.model.load_state_dict(load_weights_dict, strict=False)
                #print(self.model.load_state_dict(load_weights_dict, strict=False))
            else:
                raise FileNotFoundError("not found weights file: {}".format(self.args.weights))

    def predict(self):
        #nw = min([os.cpu_count(), self.args.batch_size if self.args.batch_size > 1 else 0, 8])  # number of workers
        nw = 32
        size = 640
        data_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.CenterCrop((size, size))])
        normalization = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        distorted_test_paths = self.distorted_test_dist['paths']
        distorted_test_labels = self.distorted_test_dist['labels']
        distorted_test_names = self.distorted_test_dist['names']

        # 实例化测试集
        distorted_test_dataset = MyDataSet_d(images_paths=distorted_test_paths,
                                             images_labels=distorted_test_labels,
                                             images_names=distorted_test_names,
                                             transform=data_transform,
                                             normalization=normalization)

        distorted_test_loader = torch.utils.data.DataLoader(distorted_test_dataset,
                                                            batch_size=self.args.batch_size,
                                                            shuffle=False,
                                                            pin_memory=False,
                                                            num_workers=nw,
                                                            collate_fn=distorted_test_dataset.collate_fn)


        test_loss, test_acc, Pred_classes = test(model=self.model,
                                                 distorted_data_loader=distorted_test_loader,
                                                 data_path=self.args.data_path,
                                                 device=self.args.device,
                                                 size=size)
        if self.args.backbone == 'Eff':
            with open(os.path.join(self.args.save_path, self.args.backbone, self.args.pred_file), 'w') as f:
                json.dump({'pre': Pred_classes}, f, indent=2)
        elif self.args.backbone == 'Res':
            with open(os.path.join(self.args.save_path, self.args.backbone, self.args.pred_file), 'w') as f:
                json.dump({'pre': Pred_classes}, f, indent=2)
        return test_loss, test_acc, Pred_classes

    def normalization_labels(self, pred_classes, window_len, T):
        if np.sum(np.abs(np.array(pred_classes[:len(pred_classes) - 1]) - np.array(
                pred_classes[1:len(pred_classes)]))) <= 1:
            pre_undistort_index = [j for j in range(len(pred_classes)) if pred_classes[j] == 0]
            if len(pre_undistort_index) == 0:
                pre_jrd = 0
            else:
                pre_jrd = max(pre_undistort_index)
            return pred_classes, pre_jrd
        else:
            qp_range = list(range(64))
            search_indexs = qp_range[::-1][window_len-1:]
            for search_index in search_indexs:
                window_value = pred_classes[search_index:search_index+window_len]
                if window_value[-1] == 0 and sum(window_value) <= T:
                    pre_jrd = search_index+window_len-1
                    return pred_classes, pre_jrd
            return pred_classes, 0

    def select_JRD(self):
        self.prepare_data()
        self.creat_model()

        # 记录开始时间
        start_time = time.time()
        test_loss, test_acc, Pred_classes = self.predict()
        # with open(os.path.join(self.args.save_path, self.args.backbone, self.args.pred_file), 'r') as f:
        #     Pred_classes = json.load(f)
        # Pred_classes = Pred_classes['pre']
        pre_jrds = []
        gt_jrds = []
        abs_delta_jrds = []
        QP_abs_dist = {k:[] for k in list(range(64))}
        for i in tqdm(range(len(self.images_names))):
            if self.args.batch_size == 64:
                pred_classes = Pred_classes[i]
            elif self.args.batch_size == 32:
                pred_classes = Pred_classes[2*i] + Pred_classes[2*i+1]
            # Pred-JRD
            pred_classes, pre_jrd = self.normalization_labels(pred_classes, window_len=8, T=5)
            pre_jrds.append(pre_jrd)
            # GT-JRD
            images_name = self.images_names[i]
            gt_jrd = self.GT_JRDs_dist[images_name]
            gt_jrds.append(gt_jrd)
            # delta
            abs_delta_jrd = abs(gt_jrd - pre_jrd)
            QP_abs_dist[gt_jrd].append(abs_delta_jrd)
            abs_delta_jrds.append(abs_delta_jrd)

        # 记录结束时间
        end_time = time.time()
        # 计算总运行时间
        time_elapsed = end_time - start_time
        print(f"Total execution time: {time_elapsed:.3f} seconds")

        # 保存数据到Excel中
        import pandas as pd
        # 创建DataFrame
        df1 = pd.DataFrame({
            'GT_classes': gt_jrds,
            'Pred_classes': pre_jrds,
            'delta': np.array(pre_jrds) - np.array(gt_jrds),
            'abs_delta_JRDs': abs_delta_jrds
        })
        with pd.ExcelWriter('pre.xlsx', engine='openpyxl') as writer:
            df1.to_excel(writer, sheet_name='Sheet1', index=False)

        print('MAE:', np.mean(abs_delta_jrds))

        # different object sizes and key categories
        objectclasses = []
        objectsizes = []
        for test_target_name in self.images_names:
            objectclass = self.objectinfo[test_target_name]['CategoryId']
            objectclasses.append(objectclass)
            test_object_path = os.path.join(self.args.data_path, 'original', test_target_name, test_target_name + '.png')
            test_object_img = cv2.imread(test_object_path)
            height = test_object_img.shape[0]
            width = test_object_img.shape[1]
            objectsizes.append(height * width)

        index_smallobject = [i for i in range(len(objectsizes)) if objectsizes[i] < 32 * 32]
        index_medobject = [i for i in range(len(objectsizes)) if objectsizes[i] >= 32 * 32 and objectsizes[i] < 96 * 96]
        index_largeobject = [i for i in range(len(objectsizes)) if objectsizes[i] >= 96 * 96]
        index_peopleobject = [i for i in range(len(objectclasses)) if objectclasses[i] == 0]
        index_carobject = [i for i in range(len(objectclasses)) if objectclasses[i] == 2]

        print('small:', len(index_smallobject),
              np.mean([abs_delta_jrds[i] for i in range(len(abs_delta_jrds)) if i in index_smallobject]))
        print('medium:', len(index_medobject),
              np.mean([abs_delta_jrds[i] for i in range(len(abs_delta_jrds)) if i in index_medobject]))
        print('large:', len(index_largeobject),
              np.mean([abs_delta_jrds[i] for i in range(len(abs_delta_jrds)) if i in index_largeobject]))
        print('people:', len(index_peopleobject),
              np.mean([abs_delta_jrds[i] for i in range(len(abs_delta_jrds)) if i in index_peopleobject]))
        print('car:', len(index_carobject),
              np.mean([abs_delta_jrds[i] for i in range(len(abs_delta_jrds)) if i in index_carobject]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--backbone', type=str, default='Eff', help='Eff, Res')
    parser.add_argument('--model_layers', type=int, default=34, help='18, 34, 50')
    parser.add_argument('--weights', type=str,
                        default='./train_weights/Eff/Eff.pth')
    parser.add_argument('--GroundTrue_JRD_path', type=str,
                        default='./jsonfiles/JRD_info.json')
    parser.add_argument('--test_info_path', type=str,
                        default='./jsonfiles/test.json')
    parser.add_argument('--objectinfo_json_path', type=str,
                        default='./jsonfiles/objects_infos.json')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--save_path', type=str, default='./predict_files')
    parser.add_argument('--pred_file', type=str, default='pred.json')
    args = parser.parse_args()
    pjrd = PJRD(args)
    pjrd.select_JRD()
