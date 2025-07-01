"""
将png格式转为yuv
"""
import os
import cv2
from tqdm import tqdm
from natsort import natsorted

resize_png_path = './data/resize_png'
yuv_path = './data/yuv'

png_file_names = list(natsorted(os.listdir(resize_png_path)))
os.chdir('./ffmpeg-4.2.2w/bin')
for png_file_name in tqdm(png_file_names):
    png_file_path = os.path.join(resize_png_path, png_file_name)
    png = cv2.imread(png_file_path)
    height = png.shape[0]
    width = png.shape[1]
    yuv_file_path = os.path.join(yuv_path, png_file_name[:-4] + '.yuv')
    cmd_png2yuv = 'ffmpeg -r 31 -i ' + png_file_path + ' -pix_fmt yuv420p -s ' + str(width) + 'x' + str(height) + ' ' + yuv_file_path
    os.system(cmd_png2yuv)