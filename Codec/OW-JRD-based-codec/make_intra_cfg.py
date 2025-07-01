"""
制作每个图像的编码参数文件，修改InitialQP参数数值
"""
import os
import json
import shutil
import numpy as np
from tqdm import tqdm
from natsort import natsorted

def changetext(filename, a,b):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = []  # 创建了一个空列表，里面没有元素
        for line in f.readlines():
            if line != '\n':
                lines.append(line)
        f.close()
    with open(filename, 'w', encoding='utf-8') as f:
        for line in lines:
            if a in line:
                line = b
                f.write('%s\n' % line)
            else:
                f.write('%s' % line)
base_QPS = {'GT': [], 'Pre': [12]}  # 选取真实的OW-JRD编码方案和预测的OW-JRD编码方案的baseQP
CTU_sizes = {'GT': 64, 'Pre': 64}   # 设置真实的OW-JRD编码方案和预测的OW-JRD编码方案的CTU大小
GTorPRE = ['GT', 'Pre']  # 是用真实的OW-JRD编码方案压缩图像还是用预测的OW-JRD编码方案压缩图像，或者是两者都进行

pre_base_path = './test_img/pre/DTJRD'  # 预测的OW-JRD编码方案的所有编码文件存放路径
gt_base_path = './test_img/gt'  # 真实的OW-JRD编码方案的所有编码文件存放路径
base_paths = {'GT': gt_base_path, 'Pre': pre_base_path}

data_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'data')
pngs_path = os.path.join(data_path, 'png')  # 待压缩图片路径，这里只用于获取图片文件名
png_names = list(natsorted(os.listdir(pngs_path)))

intra_cfg64_path = './test_img/encoder_intra_vtm64.cfg'  # CTU64×64
intra_cfg128_path = './test_img/encoder_intra_vtm128.cfg'  # CTU128×128
intra_cfgsize_paths = {'64': intra_cfg64_path, '128': intra_cfg128_path}

for gtorpre in GTorPRE:
    base_QPs = base_QPS[gtorpre]
    for base_QP in base_QPs:
        CTU_size = CTU_sizes[gtorpre]
        base_path = base_paths[gtorpre]
        for png_name in tqdm(png_names):
            file_name = 'base_QP' + str(base_QP) + '-' + 'CTUsize' + str(CTU_size) + '_0'
            pre_file_name_path = os.path.join(base_path, file_name)
            pre_file_name_intra_cfg_path = os.path.join(pre_file_name_path, 'intra_cfg')  # intra_cfg目录下存放每个图像文件的编码参数信息
            # 修改InitialQP参数为baseQP
            pre_target_intra_cfg_path = os.path.join(pre_file_name_intra_cfg_path, png_name[:-4] + '.txt')
            shutil.copy(intra_cfgsize_paths[str(CTU_size)], pre_target_intra_cfg_path)
            with open(pre_target_intra_cfg_path, 'r') as file_to_read:
                i = 1
                while True:
                    lines = file_to_read.readline()  # 整行读取数据
                    if i == 56:
                        a = lines[:37] + ' ' + str(base_QP)  # 修改InitialQP
                        changetext(pre_target_intra_cfg_path, lines, a)
                        break
                    i = i + 1
            os.rename(pre_target_intra_cfg_path, pre_target_intra_cfg_path[:-4] + '.cfg')
