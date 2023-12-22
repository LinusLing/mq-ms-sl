import os
import re
from datetime import datetime

root_path = 'D:\\PRVR_dataset'

folder_list = ['activitynet', 'charades', 'tvr']

# 遍历 folder_list 元素名称对应文件夹中的 results 文件夹
for folder in folder_list:
    folder_path = os.path.join(root_path, folder, 'results')
    time_consumption_data = []
    # 遍历 folder_path 中的所有文件夹
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        # 如果 train.log.txt 在该文件夹中，则统计耗时
        if 'train.log.txt' in os.listdir(file_path) and 'model.ckpt' in os.listdir(file_path):
            # 用于存储日志开始和结束的时间戳
            start_timestamp = None
            end_timestamp = None
            epoch_num = 0

            # 打开日志文件并逐行读取
            with open(os.path.join(file_path, 'train.log.txt'), 'r') as log_file:
                for line in log_file:
                    # 使用正则表达式匹配时间戳
                    match = re.match(r'(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})', line)
                    if match:
                        timestamp = match.group(1)
                        if start_timestamp is None:
                            start_timestamp = timestamp
                        end_timestamp = timestamp
                        epoch_num += 1

            # 将时间戳转换为datetime对象
            start_time = datetime.strptime(start_timestamp, "%Y_%m_%d_%H_%M_%S")
            end_time = datetime.strptime(end_timestamp, "%Y_%m_%d_%H_%M_%S")

            # 计算耗时
            elapsed_time = end_time - start_time
            time_consumption_data.append((file, elapsed_time, epoch_num))
    # 对 time_consumption_data 按照 file 降序排序
    time_consumption_data.sort(key=lambda x: x[0], reverse=True)
    for i, (folder_name, elapsed_time, epoch_num) in enumerate(time_consumption_data[:]):
        print(f"第 {i+1} 个：经历 {epoch_num} 个 epoch，耗时 {elapsed_time}，文件夹名字 {folder_name}")
    else:
        print('-'*30)