"""
用DecoderAnalyserApp.exe生成vtmbmsstats文件，结合YUVviewer软件观察YUV文件的CTU的QP
"""
import os
import cv2
import shutil
from tqdm import tqdm
from natsort import natsorted

target_see_base_path = 'Codec/OW-JRD-based-codec/test_img/see_CTU_QP'
file_base_path = 'Codec/OW-JRD-based-codec/test_img/pre'
if file_base_path.split('/')[-1] == 'allintraQP':
    base_QPS = [10, 15, 20, 25, 30, 35, 40, 45, 50] # this is an example, modify it by your need
else:
    base_QPS = [12]

see_yuv_name = '28170'  # this is an example, modify it by your need
target_see_path = os.path.join(target_see_base_path, see_yuv_name)  # ./test_img/see_CTU_QP/01724
if not os.path.exists(target_see_path):
    os.makedirs(target_see_path)
for base_QP in base_QPS:
    CTU_size = 64
    if file_base_path.split('/')[-1] != 'allintraQP':
        file_name = 'base_QP' + str(base_QP) + '-' + 'CTUsize' + str(CTU_size)
    else:
        file_name = str(base_QP)
    pre_file_name_path = os.path.join(file_base_path, file_name)

    target_see_file_path = os.path.join(target_see_path, 'pre-' + file_name)
    if not os.path.exists(target_see_file_path):
        os.makedirs(target_see_file_path)
    reconstruct_yuv_path = os.path.join(pre_file_name_path, 'reconstruct_yuv', see_yuv_name + '.yuv')
    target_rec_yuv_path = os.path.join(target_see_file_path, see_yuv_name + '.yuv')
    shutil.copy(reconstruct_yuv_path, target_rec_yuv_path)

    vvc_path = os.path.join(pre_file_name_path, 'bin', see_yuv_name + '.vvc')
    target_vvc_path = os.path.join(target_see_file_path, see_yuv_name + '.vvc')
    shutil.copy(vvc_path, target_vvc_path)
    target_bin_path = os.path.join(target_see_file_path, see_yuv_name + '.bin')
    os.rename(target_vvc_path, target_bin_path)

    vtmbmsstats_path = os.path.join(target_see_file_path, see_yuv_name + '.vtmbmsstats')
    cmd = 'DecoderAnalyserApp.exe -b ' + target_bin_path + ' --TraceFile="' + vtmbmsstats_path + '" --TraceRule="D_BLOCK_STATISTICS_ALL:poc>=0"' + '>'+ target_see_file_path+ '/Dtrace_out.txt'

    print(cmd)
    os.system(cmd)
