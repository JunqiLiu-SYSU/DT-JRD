"""
resize解码的png图像为原来的尺寸
"""
import os
import cv2
from tqdm import tqdm
from natsort import natsorted
from multiprocessing import Pool
base_QPS = {'GT': [], 'Pre': [12]}
CTU_sizes = {'GT': 64, 'Pre': 64}
GTorPRE = ['GT', 'Pre']

pre_base_path = './test_img/pre/DTJRD'
gt_base_path = './test_img/gt'
base_paths = {'GT': gt_base_path, 'Pre': pre_base_path}

data_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'data')
pngs_path = os.path.join(data_path, 'png')  # 待压缩图片路径，这里只用于获取图片文件名
png_names = list(natsorted(os.listdir(pngs_path)))
png_names = [png_name[:5] for png_name in png_names]
pre_Combination = []

for gtorpre in GTorPRE:
    base_QPs = base_QPS[gtorpre]
    for base_QP in base_QPs:
        CTU_size = CTU_sizes[gtorpre]
        base_path = base_paths[gtorpre]
        for png_name in tqdm(png_names):
                file_name = 'base_QP' + str(base_QP) + '-' + 'CTUsize' + str(CTU_size) + '_-1'
                png_path = os.path.join(pngs_path, png_name + '.png')
                img = cv2.imread(png_path)
                width = img.shape[1]
                height = img.shape[0]

                pre_file_name_path = os.path.join(base_path, file_name)
                decode_yuv2png_path = os.path.join(pre_file_name_path, 'decode_yuv2png', png_name + '.png')
                decode_resize_png_path = os.path.join(pre_file_name_path, 'resize_decode_png', png_name + '.png')
                pre_Combination.append([width, height, decode_yuv2png_path, decode_resize_png_path])


def decode(Combination):
    width = Combination[0]
    height = Combination[1]
    decode_yuv2png_path = Combination[2]
    decode_resize_png_path = Combination[3]
    cmd = "ffmpeg -i " + decode_yuv2png_path + " -s " + str(width) + 'x' + str(height) + ' ' + decode_resize_png_path
    print(cmd)
    os.system(cmd)

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'ffmpeg-4.2.2w', 'bin'))  # ffmpeg工具所在目录
    pool = Pool(processes=4)
    pool.map(decode, pre_Combination)
    pool.close()
    pool.join()
