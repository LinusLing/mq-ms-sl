import os

root_path = 'D:\\PRVR_dataset'

folder_list = ['activitynet', 'charades', 'tvr']

# 遍历 folder_list 元素名称对应文件夹中的 results 文件夹
for folder in folder_list:
    folder_path = os.path.join(root_path, folder, 'results')
    # 遍历 folder_path 中的所有文件夹
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        # 如果 model.ckpt 不在该文件夹中，则删除该文件夹
        if 'model.ckpt' not in os.listdir(file_path):
            # 强行删除文件夹
            os.system('rd /s /q {}'.format(file_path))
            print('remove folder: {}'.format(file_path))

