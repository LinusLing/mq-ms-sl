import re
import matplotlib.pyplot as plt
import os

# 用于存储从日志文件中提取的数据
dates = []
epochs = []
clip_nce_losses = []
clip_trip_losses = []
frame_nce_losses = []
frame_trip_losses = []
loss_overalls = []

base_path = r'D:\PRVR_dataset\charades\results\charades-run_0-2023_11_05_13_34_49'

# 打开日志文件并逐行解析
with open(os.path.join(base_path, 'train.log.txt'), 'r') as file:
    for line in file:
        # 使用正则表达式提取数据
        match = re.search(r'(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}) \[Epoch\] (\d+) \[Loss\] clip_nce_loss ([0-9.]+) clip_trip_loss ([0-9.]+) frame_nce_loss ([0-9.]+) frame_trip_loss ([0-9.]+) loss_overall ([0-9.]+)', line)
        if match:
            date, epoch, clip_nce_loss, clip_trip_loss, frame_nce_loss, frame_trip_loss, loss_overall = match.groups()
            dates.append(date)
            epochs.append(int(epoch))
            clip_nce_losses.append(float(clip_nce_loss))
            clip_trip_losses.append(float(clip_trip_loss))
            frame_nce_losses.append(float(frame_nce_loss))
            frame_trip_losses.append(float(frame_trip_loss))
            loss_overalls.append(float(loss_overall))

# 绘制曲线图
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_overalls, label='loss_overall')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.legend()
plt.grid(True)
plt.show()