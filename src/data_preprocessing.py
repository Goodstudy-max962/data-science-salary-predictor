"""
数据科学家薪资预测 - 数据预处理模块
"""
import pandas as pd
import numpy as np

# ==================== 1. 加载数据 ====================
print("正在加载数据...")
df = pd.read_csv('../data/glassdoor_jobs.csv')

print(f"原始数据形状: {df.shape}")

# ==================== 2. 数据清洗 ====================
df_clean = df.copy()

# 删除无用的第一列
if 'Unnamed: 0' in df_clean.columns:
    df_clean = df_clean.drop('Unnamed: 0', axis=1)

# 处理薪资：直接用已有的 avg_salary 列
if 'avg_salary' in df_clean.columns:
    df_clean['avg_salary_k'] = df_clean['avg_salary']
else:
    df_clean['avg_salary_k'] = (df_clean['min_salary'] + df_clean['max_salary']) / 2

# 删除薪资为 -1 或空的行
df_clean = df_clean[df_clean['avg_salary_k'] > 0]
df_clean = df_clean.dropna(subset=['avg_salary_k'])

print(f"清洗后数据形状: {df_clean.shape}")

# ==================== 3. 提取公司名 ====================
if 'Company Name' in df_clean.columns:
    df_clean['company_clean'] = df_clean['Company Name'].apply(
        lambda x: x.split('\n')[0] if isinstance(x, str) else str(x)
    )

# ==================== 4. 地区信息 ====================
if 'job_state' not in df_clean.columns and 'Location' in df_clean.columns:
    df_clean['job_state'] = df_clean['Location'].apply(
        lambda x: x.split(',')[-1].strip() if isinstance(x, str) else x
    )

# ==================== 5. 技能特征 ====================
skill_cols_in_data = ['python_yn', 'R_yn', 'spark', 'aws', 'excel']

if 'Job Description' in df_clean.columns:
    extra_skills = ['sql', 'tableau', 'power bi', 'hadoop', 'azure',
                    'tensorflow', 'pytorch', 'machine learning', 'deep learning']
    for skill in extra_skills:
        col_name = f'skill_{skill.replace(" ", "_")}'
        df_clean[col_name] = df_clean['Job Description'].apply(
            lambda x: 1 if isinstance(x, str) and skill.lower() in x.lower() else 0
        )

# ==================== 6. 选择最终特征列 ====================
base_cols = ['avg_salary_k', 'Location', 'job_state', 'Industry', 'Sector',
             'Revenue', 'Size', 'Type of ownership', 'company_clean',
             'Rating', 'age', 'num_comp', 'seniority', 'desc_len']

base_cols.extend(skill_cols_in_data)

extra_skill_cols = [f'skill_{s.replace(" ", "_")}' for s in
                    ['sql', 'tableau', 'power_bi', 'hadoop', 'azure',
                     'tensorflow', 'pytorch', 'machine_learning', 'deep_learning']]
base_cols.extend(extra_skill_cols)

existing_cols = [col for col in base_cols if col in df_clean.columns]
df_final = df_clean[existing_cols].copy()

# ==================== 7. 处理缺失值 ====================
for col in df_final.columns:
    if df_final[col].dtype == 'object':
        df_final[col] = df_final[col].fillna('Unknown')
    else:
        df_final[col] = df_final[col].fillna(0)

df_final = df_final[df_final['avg_salary_k'] > 0]

print(f"\n最终数据形状: {df_final.shape}")
print(f"特征列数: {len(df_final.columns)}")

# ==================== 8. 保存处理后的数据 ====================
output_path = '../data/processed_data.csv'
df_final.to_csv(output_path, index=False)
print(f"\n数据预处理完成！已保存到: {output_path}")
print("可以运行 model_training.py 开始建模")