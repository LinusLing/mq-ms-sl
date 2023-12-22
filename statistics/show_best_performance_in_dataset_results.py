import os
import re
root_path = 'D:\\PRVR_dataset'

dataset_name_list = ['activitynet', 'charades', 'tvr']
performance_baseline_list = [140.1, 68.4, 172.3]

for i, dataset_name in enumerate(dataset_name_list):

    base_path = os.path.join(root_path, dataset_name, 'results')

    # 用于存储所有性能数据和对应的文件夹名字
    performance_data = []

    # 遍历base_path下的所有文件夹
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        log_file_path = os.path.join(folder_path, "performance.log")

        # 检查文件是否存在
        if os.path.isfile(log_file_path):
            with open(log_file_path, 'r') as log_file:
                log_data = log_file.read()

                # 使用正则表达式从后往前搜索匹配的“recall sum”数值
                matches = re.findall(r'recall sum: (\d+\.\d+)', log_data)

                if matches:
                    # 只保留最后一个匹配
                    recall_sum = float(matches[-1])
                    performance_data.append((folder_name, recall_sum))

    # 对性能数据按照性能值降序排序
    performance_data.sort(key=lambda x: x[1], reverse=True)

    # 打印前三名性能和对应的文件夹名字
    if performance_data:
        print(f"{dataset_name} baseline: {performance_baseline_list[i]} 前三名性能数据和对应的文件夹名字：")
        for i, (folder_name, recall_sum) in enumerate(performance_data[:]):
            print(f"第 {i+1} 名：性能 {recall_sum}，文件夹名字 {folder_name}")
        else:
            print('-'*30)
    else:
        print("未找到任何性能数据。")