"""
为每种不同参数的编码方案制作存放编码信息的文件夹
"""
import os
base_QPS = {'GT': [], 'Pre': [12]}  # 选取真实的OW-JRD编码方案和预测的OW-JRD编码方案的baseQP
CTU_sizes = {'GT': 64, 'Pre': 64}   # 设置真实的OW-JRD编码方案和预测的OW-JRD编码方案的CTU大小
GTorPRE = ['GT', 'Pre']  # 是用真实的OW-JRD编码方案压缩图像还是用预测的OW-JRD编码方案压缩图像，或者是两者都进行

pre_base_path = './test_img/pre/DTJRD'  # 预测的OW-JRD编码方案的所有编码文件存放路径
gt_base_path = './test_img/gt'  # 真实的OW-JRD编码方案的所有编码文件存放路径
base_paths = {'GT': gt_base_path, 'Pre': pre_base_path}

for gtorpre in GTorPRE:
    base_QPs = base_QPS[gtorpre]
    for base_QP in base_QPs:
        CTU_size = CTU_sizes[gtorpre]
        base_path = base_paths[gtorpre]

        file_name = 'base_QP' + str(base_QP) + '-' + 'CTUsize' + str(CTU_size) + '_0'
        pre_file_name_path = os.path.join(base_path, file_name)
        os.makedirs(pre_file_name_path)

        pre_file_name_intra_cfg_path = os.path.join(pre_file_name_path, 'intra_cfg')
        os.makedirs(pre_file_name_intra_cfg_path)

        pre_file_name_bin_path = os.path.join(pre_file_name_path, 'bin')
        os.makedirs(pre_file_name_bin_path)

        pre_file_name_decode_yuv_path = os.path.join(pre_file_name_path, 'decode_yuv')
        os.makedirs(pre_file_name_decode_yuv_path)

        pre_file_name_decode_yuv2png_path = os.path.join(pre_file_name_path, 'decode_yuv2png')
        os.makedirs(pre_file_name_decode_yuv2png_path)

        pre_file_name_Enc_out_path = os.path.join(pre_file_name_path, 'Enc_out')
        os.makedirs(pre_file_name_Enc_out_path)

        pre_file_name_reconstruct_yuv_path = os.path.join(pre_file_name_path, 'reconstruct_yuv')
        os.makedirs(pre_file_name_reconstruct_yuv_path)

        pre_file_name_resize_decode_png_path = os.path.join(pre_file_name_path, 'resize_decode_png')
        os.makedirs(pre_file_name_resize_decode_png_path)
