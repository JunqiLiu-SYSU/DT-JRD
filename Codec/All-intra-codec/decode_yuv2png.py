import os
import cv2
from tqdm import tqdm
from natsort import natsorted
from multiprocessing import Pool

QPs = [26,27,28,29,31,32,33,34] # this is an example, modify it by your need
data_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'data')
resize_png_paths = os.path.join(data_path, 'resize_png')
png_names = list(natsorted(os.listdir(resize_png_paths)))
png_names = [png_name[:5] for png_name in png_names]
Combination = []
for QP in QPs:
    for png_name in tqdm(png_names):
        png_path = os.path.join(resize_png_paths, png_name + '.png')
        img = cv2.imread(png_path)
        width = img.shape[1]
        height = img.shape[0]
        decode_yuv_path = 'C:/Users/Administrator.DESKTOP-4KE2HTR/Desktop/Codec/All-intra-codec/test_img/allintraQP/' + str(QP) + '/decode_yuv/' + png_name + '.yuv'
        decode_png_path = 'C:/Users/Administrator.DESKTOP-4KE2HTR/Desktop/Codec/All-intra-codec/test_img/allintraQP/' + str(QP) + '/decode_yuv2png/' + png_name + '.png'
        Combination.append([width, height, decode_yuv_path, decode_png_path])

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
    pool.map(decode, Combination)
    pool.close()
    pool.join()
