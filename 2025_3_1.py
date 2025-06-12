import math

file_path = r'"G:\ur_angle_trajectory\angles_list.txt"'

# 读取文件
with open(file_path, 'r') as file:
    lines = file.readlines()

# 初始化一个空列表，用于存储所有数据
data = []

# 处理每一行数据
for line in lines:
    # 去除中括号和换行符
    line = line.strip()[1:-1]
    # 将字符串分割为单独的数字
    values = list(map(float, line.split()))
    # 将角度转换为弧度
    radians = [math.radians(angle) for angle in values]
    # 将转换后的弧度值添加到数据列表中
    data.append(radians)

# 打印结果
for i, row in enumerate(data):
    print(f"第 {i+1} 行弧度值: {row}")