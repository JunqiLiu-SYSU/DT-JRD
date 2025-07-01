"""
制作每个图像的信息配置文件
"""
import os
import cv2
import shutil
from natsort import natsorted

def changetext(filename, a,b):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = []
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

video_param_txt_path = './data/video_param.txt'
cfg_path = './data/png_cfg'
pngs_path = './data/resize_png'
yuvs_path = './data/yuv/'

png_names = list(natsorted(os.listdir(pngs_path)))
for png_name in png_names:
    png_path = os.path.join(pngs_path, png_name)
    png = cv2.imread(png_path)
    height = png.shape[0]
    width = png.shape[1]
    target_yuv_param_txt_path = os.path.join(cfg_path, png_name[:-4] + '.txt')
    shutil.copy(video_param_txt_path, target_yuv_param_txt_path)
    with open(target_yuv_param_txt_path, 'r') as file_to_read:
        i = 1
        while True:
            lines = file_to_read.readline()
            if i == 2:
                a = lines[:32] + yuvs_path + png_name[:-4] + '.yuv'
                changetext(target_yuv_param_txt_path, lines, a)
            elif i == 7:
                b = lines[:32] + str(width)
                changetext(target_yuv_param_txt_path, lines, b)
            elif i == 8:
                c = lines[:32] + str(height)
                changetext(target_yuv_param_txt_path, lines, c)
                break
            i = i + 1
            if not lines:
                break

    os.rename(target_yuv_param_txt_path, target_yuv_param_txt_path[:-4] + '.cfg')
