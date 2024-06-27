import pandas as pd

# 读取Parquet文件
parquet_file = './advbench.parquet'
df = pd.read_parquet(parquet_file)

# 将DataFrame保存为CSV文件
save_name = parquet_file.split('/')[-1].split('.')[0]
csv_file = f'{save_name}.csv'
df.to_csv(csv_file, index=False)

print(f"Parquet文件已成功转换为CSV文件：{csv_file}")