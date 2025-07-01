"""
Yuv格式转png
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
resize_png_paths = os.path.join(data_path, 'resize_png')  # 待压缩图片的信息配置文件路径
png_names = list(natsorted(os.listdir(resize_png_paths)))
png_names = [png_name[:5] for png_name in png_names]
pre_Combination = []

for gtorpre in GTorPRE:
    base_QPs = base_QPS[gtorpre]
    for base_QP in base_QPs:
        CTU_size = CTU_sizes[gtorpre]
        base_path = base_paths[gtorpre]
        for png_name in tqdm(png_names):
            file_name = 'base_QP' + str(base_QP) + '-' + 'CTUsize' + str(CTU_size) + '_-1'
            png_path = os.path.join(resize_png_paths, png_name + '.png')
            img = cv2.imread(png_path)
            width = img.shape[1]
            height = img.shape[0]

            pre_file_name_path = os.path.join(base_path, file_name)
            decode_yuv_path = os.path.join(pre_file_name_path, 'decode_yuv', png_name + '.yuv')
            decode_png_path = os.path.join(pre_file_name_path, 'decode_yuv2png', png_name + '.png')
            pre_Combination.append([width, height, decode_yuv_path, decode_png_path])


def decode(Combination):
    width = Combination[0]
    height = Combination[1]
    decode_yuv_path = Combination[2]
    decode_png_path = Combination[3]
    cmd = 'ffmpeg -pix_fmt yuv420p -f rawvideo -s ' + str(width) + 'x' + str(height) + ' -i ' + decode_yuv_path + ' ' + decode_png_path
    print(cmd)
    os.system(cmd)

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'ffmpeg-4.2.2w', 'bin'))  # ffmpeg工具所在目录
    pool = Pool(processes=4)
    pool.map(decode, pre_Combination)
    pool.close()
    pool.join()

