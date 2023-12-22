import subprocess

# 定义要运行的文件名
file_name = '../method/train.py'

# 循环运行十次
for _ in range(3):
    # 使用subprocess模块运行python文件
    subprocess.run(['python', file_name])
