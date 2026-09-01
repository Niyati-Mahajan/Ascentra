import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from app.config import settings

def generate_eda(df: pd.DataFrame) -> str:
    eda_dir = os.path.join(settings.BASE_DIR, "reports", "eda")
    os.makedirs(eda_dir, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    # 1. Target Distribution Plot
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='PlacementStatus', palette='viridis')
    plt.title('Target Distribution: Placement Status')
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, 'placement_status_distribution.png'))
    plt.close()
    
    # 2. CGPA vs Placement Status
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x='PlacementStatus', y='CGPA', palette='Set2')
    plt.title('CGPA Distribution by Placement Status')
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, 'cgpa_vs_placement.png'))
    plt.close()
    
    # 3. Internships vs Placement Status
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x='Internships', hue='PlacementStatus', palette='coolwarm')
    plt.title('Internship Experience vs Placement Status')
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, 'internships_vs_placement.png'))
    plt.close()
    
    # 4. Correlation Heatmap
    plt.figure(figsize=(9, 7))
    num_df = df.select_dtypes(include=[np.number]).copy()
    num_df['Placement_Numeric'] = (df['PlacementStatus'] == 'Placed').astype(int)
    corr = num_df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, 'correlation_matrix.png'))
    plt.close()
    
    return eda_dir
