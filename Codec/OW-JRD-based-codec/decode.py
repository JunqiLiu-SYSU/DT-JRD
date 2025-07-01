"""
解码图像
"""
import os
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
pre_need_decode_Combination = []

for gtorpre in GTorPRE:
    base_QPs = base_QPS[gtorpre]
    for base_QP in base_QPs:
        CTU_size = CTU_sizes[gtorpre]
        base_path = base_paths[gtorpre]
        for png_name in tqdm(png_names):
            file_name = 'base_QP' + str(base_QP) + '-' + 'CTUsize' + str(CTU_size) + '_0'
            pre_file_name_path = os.path.join(base_path, file_name)
            decode_yuv_path = os.path.join(pre_file_name_path, 'decode_yuv', png_name + '.yuv')
            bin_path = os.path.join(pre_file_name_path, 'bin', png_name + '.vvc')
            pre_need_decode_Combination.append([bin_path, decode_yuv_path])


def decode(Combination):
    bin_path = Combination[0]
    decode_yuv_path = Combination[1]
    cmd = 'DecoderApp.exe -b ' + bin_path + ' -o ' + decode_yuv_path
    print(cmd)
    os.system(cmd)

if __name__ == "__main__":
    # os.chdir(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'test_img'))  # 该目录文件下放置DecoderApp.exe
    pool = Pool(processes=8)
    pool.map(decode, pre_need_decode_Combination)
    pool.close()
    pool.join()