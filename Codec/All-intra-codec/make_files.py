import os

path = './test_img/allintraQP'
for i in range(10, 51, 5):  # this is an example, modify it by your need
    QP_path = os.path.join(path, str(i))
    if not os.path.exists(QP_path):
        os.makedirs(QP_path)
    QP_bin_path = os.path.join(QP_path, 'bin')
    QP_decode_yuv_path = os.path.join(QP_path, 'decode_yuv')
    QP_decode_yuv2png_path = os.path.join(QP_path, 'decode_yuv2png')
    QP_Enc_out_path = os.path.join(QP_path, 'Enc_out')
    QP_reconstruct_yuv_path = os.path.join(QP_path, 'reconstruct_yuv')
    QP_resize_decode_png_path = os.path.join(QP_path, 'resize_decode_png')
    os.makedirs(QP_bin_path)
    os.makedirs(QP_decode_yuv_path)
    os.makedirs(QP_decode_yuv2png_path)
    os.makedirs(QP_Enc_out_path)
    os.makedirs(QP_reconstruct_yuv_path)
    os.makedirs(QP_resize_decode_png_path)