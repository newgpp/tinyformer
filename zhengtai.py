import torch
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 准备数据
x = torch.randn(2, 4, 8)

print("x[0]", x[0])

data = x[0].numpy().flatten() # 取出第1个矩阵并展平为一维数组

print(data)

# 2. 绘制分布曲线
plt.figure(figsize=(8, 5))
sns.kdeplot(data, fill=True, color="blue", bw_adjust=0.5) # fill=True 会填充曲线下方颜色

# 3. 添加参考线：均值 0
plt.axvline(0, color='red', linestyle='--', label='Mean=0')

plt.title("Distribution of values in x[0] (KDE Plot)")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()
