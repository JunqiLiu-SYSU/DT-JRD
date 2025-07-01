"""
将宽高不是偶数的图像resize成为偶数值，以便于转Yuv420
"""
import os
import shutil
import cv2
from tqdm import tqdm
from natsort import natsorted

png_path = './data/png'
resize_png_path = './data/resize_png'
if not os.path.isdir(resize_png_path):
    os.makedirs(resize_png_path)
png_file_names = list(natsorted(os.listdir(png_path)))

os.chdir('./ffmpeg-4.2.2w/bin')  # ffmpeg工具所在目录
for png_file_name in tqdm(png_file_names):
    png_file_path = os.path.join(png_path, png_file_name)
    resize_png_file_path = os.path.join(resize_png_path, png_file_name)
    png = cv2.imread(png_file_path)
    height = png.shape[0]
    width = png.shape[1]
    if (width % 2 == 0) & (height % 2 == 0):
        shutil.copy(png_file_path, resize_png_file_path)
    else:
        new_height = height + height % 2
        new_width = width + width % 2
        cmd_resize_img = "ffmpeg -i " + png_file_path + " -s " + str(new_width) + 'x' + str(new_height) + ' ' + resize_png_file_path
        os.system(cmd_resize_img)