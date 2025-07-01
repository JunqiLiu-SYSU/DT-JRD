"""
针对不同编码方案的目标OW-JRD、bbox和InitialQP，分别利用多核CPU进行多进程编码
"""
import os
import json
from tqdm import tqdm
from natsort import natsorted
from multiprocessing import Pool
import time
base_QPS = {'GT': [], 'Pre': [12]}  # 选取真实的OW-JRD编码方案和预测的OW-JRD编码方案的baseQP
CTU_sizes = {'GT': 64, 'Pre': 64}   # 设置真实的OW-JRD编码方案和预测的OW-JRD编码方案的CTU大小
GTorPRE = ['GT', 'Pre']  # 是用真实的OW-JRD编码方案压缩图像还是用预测的OW-JRD编码方案压缩图像，或者是两者都进行

pre_base_path = './test_img/pre/DTJRD'  # 预测的OW-JRD编码方案的所有编码文件存放路径
gt_base_path = './test_img/gt'  # 真实的OW-JRD编码方案的所有编码文件存放路径
base_paths = {'GT': gt_base_path, 'Pre': pre_base_path}

Pre_objects_path = './test_img/DTJRD_Pre_img_objects.json'  # 该json存放每张图像中的目标预测得到的OW-JRD和目标所在位置bbox   ***需要更新json中预测的JRD值
GT_objects_path = './test_img/GTJRD_img_objects.json'  # 该json存放每张图像中的目标真实的OW-JRD和目标所在位置bbox
with open(Pre_objects_path, 'r') as f:
    Pre_objects_dist = json.load(f)
with open(GT_objects_path, 'r') as f:
    GT_objects_dist = json.load(f)

mean_error = 0  # OWJRD平均预测误差，用于减少整体预测误差带给编码的影响
data_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'data')

png_cfgs_path = os.path.join(data_path, 'png_cfg')  # 待压缩图片的信息配置文件路径
pngs_path = os.path.join(data_path, 'png')  # 待压缩图片路径，这里只用于获取图片文件名      *** 为什么这里不是'resize_png'而是'png'?没事了，用的cfg文件里是resize后的yuv（和全帧内用的yuv是一样的）
png_names = list(natsorted(os.listdir(pngs_path)))
png_names = [png_name[:5] for png_name in png_names]
pre_need_compress_Combination = []

log_file = "encoding_log.txt"
def log_message(message):
    with open(log_file, "a") as f:
        f.write(message + "\n")
    print(message)

for gtorpre in GTorPRE:
    base_QPs = base_QPS[gtorpre]
    for base_QP in base_QPs:
        CTU_size = CTU_sizes[gtorpre]
        base_path = base_paths[gtorpre]
        for png_name in tqdm(png_names):
            file_name = 'base_QP' + str(base_QP) + '-' + 'CTUsize' + str(CTU_size) + '_0'
            pre_file_name_path = os.path.join(base_path, file_name)
            intra_cfg_path = os.path.join(pre_file_name_path, 'intra_cfg', png_name + '.cfg')  # 第一个CTU的QP值的设置：更改配置cfg文件中对应Rate control的InitialQP
            cfg_path = png_cfgs_path + '/' + png_name + '.cfg'  # 待压缩的图像yuv文件信息
            reconstruct_yuv_path = os.path.join(pre_file_name_path, 'reconstruct_yuv', png_name + '.yuv')
            bin_path = os.path.join(pre_file_name_path, 'bin', png_name + '.vvc')
            Enc_out_txt_path = os.path.join(pre_file_name_path, 'Enc_out', png_name + '.txt')  # 编码信息输出文件
            bboxs = Pre_objects_dist[png_name + '.png']['bboxs']  # 获取当前图片中所有目标的bbox，编码器根据每个bbox对每个编码块进行区分为目标块和背景块，从而分配不同的QP
            if gtorpre == 'Pre':
                OWJRD = Pre_objects_dist[png_name + '.png']['PreOWJRD']
                OWJRD = [OJ + mean_error for OJ in OWJRD]  # 将预测的OW-JRD整体加上跟真实OW-JRD的误差（目标图像多的话结合模型可以作为经验值），使得预测的OW-JRD更加接近真实的OW-JRD
            else:
                OWJRD = GT_objects_dist[png_name + '.png']['OWJRD']
                OWJRD = [OJ + mean_error for OJ in OWJRD]
            bboxs = [str(i) for item in bboxs for i in item]
            OWJRD = [str(i) for i in OWJRD]
            string_bboxs = ','.join(bboxs)  # 将bbox的列表格式转为符合编码器输入的字符串格式
            string_OWJRD = ','.join(OWJRD) + ',' + str(51)  # 格式转化同上，背景用大QP，如51
            pre_need_compress_Combination.append(
                [intra_cfg_path, cfg_path, reconstruct_yuv_path, bin_path, str(base_QP), string_bboxs,
                 string_OWJRD, Enc_out_txt_path])  # 将需要编码的文件所需的所有信息合并成一个二维列表


def code(Combination):
    # 获取当前编码的文件的所有信息
    intra_cfg_path = Combination[0]
    cfg_path = Combination[1]
    reconstruct_yuv_path = Combination[2]
    bin_path = Combination[3]
    base_QP = Combination[4]        # ***疑似EncoderAPP.exe里面有脏东西，人为给初始QP -2 了
    string_bboxs = Combination[5]
    string_OWJRD = Combination[6]
    Enc_Out_txt_path = Combination[7]
    # 将信息写成cmd命令，执行编码操作
    cmd = 'EncoderApp.exe -c ' + intra_cfg_path + ' -c ' + cfg_path + ' -o ' + reconstruct_yuv_path + ' -b ' + bin_path + ' -q ' + str(base_QP) + ' ' + string_bboxs + ' ' + string_OWJRD + '   > ' + Enc_Out_txt_path
    print(cmd)
    log_message(cmd)
    os.system(cmd)

if __name__ == "__main__":
    # os.chdir(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'test_img'))  # 该目录文件下放置EncoderApp.exe
    start_time = time.time()
    log_message("Encoding process started...")
    pool = Pool(processes=8)  # CPU8个核
    pool.map(code, pre_need_compress_Combination)
    pool.close()
    pool.join()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    execution_time = f"Total execution time: {end_time - start_time:.2f} seconds"
    log_message(execution_time)