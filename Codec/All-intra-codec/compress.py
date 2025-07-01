import os
from tqdm import tqdm
from natsort import natsorted
from multiprocessing import Pool

data_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'data')
png_cfgs_path = os.path.join(data_path, 'png_cfg')
png_path = os.path.join(data_path, 'png')
png_names = list(natsorted(os.listdir(png_path)))
png_names = [png_name[:5] for png_name in png_names]
need_compress_Combination = []

intra_cfg_path = './encoder_intra_vtm.cfg'
QPs = range(10, 51, 5)  # this is an example, modify it by your need

for QP in QPs:
    for png_name in tqdm(png_names):
        cfg_path = png_cfgs_path + '/' + png_name + '.cfg'
        reconstruct_yuv_path = './test_img/allintraQP/' + str(QP) + '/reconstruct_yuv/' + png_name + '.yuv'
        bin_path = './test_img/allintraQP/' + str(QP) + '/bin/' + png_name + '.vvc'
        Enc_Out_txt_path = './test_img/allintraQP/' + str(QP) + '/Enc_out/' + png_name + '.txt'
        need_compress_Combination.append([cfg_path, reconstruct_yuv_path, bin_path, str(QP), Enc_Out_txt_path])

def code(Combination):
    cfg_path = Combination[0]
    reconstruct_yuv_path = Combination[1]
    bin_path = Combination[2]
    qp = Combination[3]
    Enc_Out_txt_path = Combination[4]
    cmd = 'EncoderApp.exe -c ' + intra_cfg_path + ' -c ' + cfg_path + ' -o ' + reconstruct_yuv_path + ' -b ' + bin_path + ' -q ' + qp + '   > ' + Enc_Out_txt_path
    print(cmd)
    os.system(cmd)

if __name__ == "__main__":
    # os.chdir(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'test_img'))
    pool = Pool(processes=4)
    pool.map(code, need_compress_Combination)
    pool.close()
    pool.join()