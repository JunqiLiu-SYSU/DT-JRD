import os
from tqdm import tqdm
from natsort import natsorted
from multiprocessing import Pool

QPs = [26,27,28,29,31,32,33,34] # this is an example, modify it by your need
data_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'data')
pngs_path = os.path.join(data_path, 'png')  # 待压缩图片路径，这里只用于获取图片文件名
png_names = list(natsorted(os.listdir(pngs_path)))
png_names = [png_name[:5] for png_name in png_names]
need_decode_Combination = []

for QP in QPs:
    for png_name in tqdm(png_names):
        bin_path = './test_img/allintraQP/' + str(QP) + '/bin/' + png_name + '.vvc'
        decode_yuv_path = './test_img/allintraQP/' + str(QP) + '/decode_yuv/' + png_name + '.yuv'
        need_decode_Combination.append([bin_path, decode_yuv_path])


def decode(Combination):
    bin_path = Combination[0]
    decode_yuv_path = Combination[1]
    cmd = 'DecoderApp.exe -b ' + bin_path + ' -o ' + decode_yuv_path
    print(cmd)
    os.system(cmd)

if __name__ == "__main__":
    # os.chdir(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'test_img'))  # 该目录文件下放置DecoderApp.exe
    pool = Pool(processes=4)
    pool.map(decode, need_decode_Combination)
    pool.close()
    pool.join()