import matplotlib.pyplot as plt
import re
import csv
# 文件路径
file_path = 'imdb_sft_gpt2large2.txt'

# 初始化一个空列表来存储 Train Loss 值
train_losses = []

# # 打开并读取文件
# with open(file_path, 'r') as file:
#     for line in file:
#         # 使用正则表达式提取 Train Loss 的数值
#         match = re.search(r'Train Loss: ([0-9.]+)', line)
#         if match:
#             train_loss = float(match.group(1))
#             train_losses.append(train_loss)

# # 使用 matplotlib 绘制折线图
# plt.figure(figsize=(10, 6))
# plt.plot(train_losses, linestyle='-', color='b')
# plt.title('Train Loss Over Time')
# plt.xlabel('Step')
# plt.ylabel('Train Loss')
# plt.grid(True)
# plt.savefig('train_loss.png')


# 输出文件路径
output_file_path = 'output.csv'

# 初始化一个空列表来存储提取的数值
data = []

# 打开并读取文件
with open(file_path, 'r') as file:
    for line in file:
        # 使用正则表达式提取 Positive Avg Score, Positive Num, Negative Score, Neg Num
        match = re.search(r'Positive Avg Score: ([0-9.]+), Positive Num: (\d+), Negative Score: ([0-9.]+), Neg Num: (\d+)', line)
        if match:
            positive_avg_score = float(match.group(1))
            positive_num = int(match.group(2))
            negative_score = float(match.group(3))
            neg_num = int(match.group(4))
            data.append([positive_avg_score, positive_num, negative_score, neg_num])

_data = []
sum4row = []
for i, row in enumerate(data):
    if i !=0 and i % 4 == 0:
        row0 = sum4row[0]
        row1 = sum4row[1]
        row2 = sum4row[2]
        row3 = sum4row[3]
        pos_num = row0[1] + row1[1] + row2[1] + row3[1]
        neg_num = row0[3] + row1[3] + row2[3] + row3[3]
        pos_avg_score = (row0[0] * row0[1] + row1[0] * row1[1] + row2[0] * row2[1] + row3[0] * row3[1]) / pos_num
        neg_avg_score = (row0[2] * row0[3] + row1[2] * row1[3] + row2[2] * row2[3] + row3[2] * row3[3]) / neg_num
        _data.append([pos_avg_score, pos_num, neg_avg_score, neg_num])
        sum4row = []
    sum4row.append(row)        


# 将提取到的数据写入 CSV 文件
with open(output_file_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # 写入表头
    csvwriter.writerow(['Positive Avg Score', 'Positive Num', 'Negative Score', 'Neg Num'])
    # 写入数据
    csvwriter.writerows(_data)

print(f"数据已成功写入 {output_file_path}")