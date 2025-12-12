import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', 80)
pd.set_option('display.max_rows', 50)

# assumptions: interest rate, term, installment are only known after approval decision
# risk score is given before (credit score)

# ## 1. Import Dataset / Clean

df_loans = pd.read_csv('cc_loan_reduced.csv')
df_og = df_loans.copy(deep = True)

print(df_loans['record_id'].nunique())
print(df_loans.shape)
df_loans.head(8)

# Check which columns are not numeric
# Expected: 
## issue_d (date)
## loan_status (string/cat): ['Charged Off', 'Fully Paid']
## last_pymnt_d (date)
## emp_title (string): job title
## emp_length (string/cat): length of employment
## home_ownership (string/cat): ['RENT', 'MORTGAGE', 'OWN']
## verification_status: whether income was verified
## purpose (string/cat)
## title (string): title provider by borrower
## zip_code (string/cat): only first three numbers, general area
## addr_state (string/cat): state
## earliest_cr_line (date): earliest credit line opened

[col for col in df_loans.columns if not pd.api.types.is_numeric_dtype(df_loans[col])]

# Fix zip_code (there are some that have few rows)

zip_counts = df_loans['zip_code'].value_counts()

threshold = 100 
rare_zips = zip_counts[zip_counts < threshold].index
df_loans['zip_code'] = df_loans['zip_code'].apply(
    lambda x: 'Other' if x in rare_zips else x
)

# Correct type for columns, identify which ones have less predictive power or shouldn/t be used
## term: all the loans have the same term: 36 months

cols_ignore = ['record_id', 'term', 'issue_d', 'last_pymnt_d', 'loan_status', 'emp_title', 'title', 
               'avg_cur_bal', 'total_pymnt', 'installment', 'int_rate']


# Create binary target
df_loans['default'] = (df_loans['loan_status'] == 'Charged Off').astype(int)
cols_ignore += ['default']

# Check missingness

(df_loans[[col for col in df_loans.columns if col not in cols_ignore]].isnull().sum() / 
 len(df_loans) * 100).sort_values(ascending = False).head(20)

# ## 2. Feature Engineering

# Let's deal with the missing variables

df_loans['has_public_record'] = df_loans['mths_since_last_record'].notna().astype(int)
cols_ignore += ['mths_since_last_record']
df_loans['has_public_record'].value_counts()

df_loans['has_bankcard_delinq'] = df_loans['mths_since_recent_bc_dlq'].notna().astype(int)
cols_ignore += ['mths_since_recent_bc_dlq']
df_loans['has_bankcard_delinq'].value_counts()

df_loans['has_major_derog'] = df_loans['mths_since_last_major_derog'].notna().astype(int)
cols_ignore += ['mths_since_last_major_derog']
df_loans['has_major_derog'].value_counts()

df_loans['has_revol_delinq'] = df_loans['mths_since_recent_revol_delinq'].notna().astype(int)
cols_ignore += ['mths_since_recent_revol_delinq']
df_loans['has_revol_delinq'].value_counts()

df_loans['has_any_delinq'] = df_loans['mths_since_last_delinq'].notna().astype(int)
cols_ignore += ['mths_since_last_delinq']
df_loans['has_any_delinq'].value_counts()

df_loans['mths_since_recent_inq'] = df_loans['mths_since_recent_inq'].fillna(np.nanmedian(df_loans['mths_since_recent_inq']))
df_loans['mths_since_recent_inq'].value_counts()

df_loans['emp_length'] = df_loans['emp_length'].fillna(df_loans['emp_length'].mode()[0])
df_loans['emp_length'] = df_loans['emp_length'].str.replace(' years', '').str.replace(' year', '')
df_loans['emp_length'].value_counts(dropna = False)

df_loans['num_tl_120dpd_2m'] = df_loans['num_tl_120dpd_2m'].fillna(np.nanmedian(df_loans['num_tl_120dpd_2m']))
df_loans['num_tl_120dpd_2m'].value_counts()

df_loans['mo_sin_old_il_acct'] = df_loans['mo_sin_old_il_acct'].fillna(np.nanmedian(df_loans['mo_sin_old_il_acct']))
df_loans['mo_sin_old_il_acct'].value_counts()

df_loans['bc_util'] = df_loans['bc_util'].fillna(np.nanmedian(df_loans['bc_util']))
df_loans['bc_util'].value_counts()

df_loans['percent_bc_gt_75'] = df_loans['percent_bc_gt_75'].fillna(np.nanmedian(df_loans['percent_bc_gt_75']))
df_loans['percent_bc_gt_75'].value_counts()

df_loans['bc_open_to_buy'] = df_loans['bc_open_to_buy'].fillna(np.nanmedian(df_loans['bc_open_to_buy']))
df_loans['bc_open_to_buy'].value_counts()

df_loans['mths_since_recent_bc'] = df_loans['mths_since_recent_bc'].fillna(np.nanmedian(df_loans['mths_since_recent_bc']))
df_loans['mths_since_recent_bc'].value_counts()

df_loans['revol_util'] = df_loans['revol_util'].fillna(np.nanmedian(df_loans['revol_util']))
df_loans['revol_util'].value_counts()

# Check missingness

(df_loans[[col for col in df_loans.columns if col not in cols_ignore]].isnull().sum() / 
 len(df_loans) * 100).sort_values(ascending = False).head(20)

# Default rate
print(f"Overall default rate: {df_loans['default'].mean():.2%}")
print(f"Number of defaults: {df_loans[df_loans['default'] == 1].shape[0]}")

# Default rate over time
df_loans.groupby('issue_d')['default'].mean().plot()
plt.title('Default Rate by Issue Date')

df_loans['revol_util'].hist()

## Other features

# Income bins (log)
df_loans['income_bin'] = pd.cut(df_loans['annual_inc'], 
                                 bins=[0, 30000, 50000, 60000, 75000, 100000, np.inf],
                                 labels=['<30k', '30-50k', '50-60k', '60-75k', '75-100k', '100k+'])
cols_ignore += ['annual_inc']

# DTI bins 
df_loans['dti_bin'] = pd.cut(df_loans['dti'],
                              bins=[0, 5, 10, 15, 20, 25, 30, 35,np.inf],
                              labels=['<5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35+'])
cols_ignore += ['dti']

# Loan amount bins
df_loans['loan_amnt_bin'] = pd.qcut(df_loans['loan_amnt'], 
                                     q=5, 
                                     labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
cols_ignore += ['loan_amnt']

# Revolving utilization bins 
df_loans['revol_util_bin'] = pd.cut(df_loans['revol_util'],
                                     bins=[0, 30, 50, 60, 70, 80, 90, np.inf],
                                     labels=['<30%', '30-50%', '50-60%', '60-70%', '70-80%', '80-90%', '>90%'])
cols_ignore += ['revol_util']

df_loans['revol_util_bin'].value_counts()

# Credit History
df_loans['issue_d'] = pd.to_datetime(df_loans['issue_d'], format = '%b-%Y')
df_loans['earliest_cr_line'] = pd.to_datetime(df_loans['earliest_cr_line'], format = '%b-%Y')
    
df_loans['credit_history_months'] = (df_loans['issue_d'] - df_loans['earliest_cr_line']).dt.days / 30.44
cols_ignore += ['earliest_cr_line']
df_loans['credit_history_months'].hist()

'''
# Income to loan ratio
df_loans['income_to_loan'] = df_loans['annual_inc'] / df_loans['loan_amnt']
'''

# Available credit
df_loans['available_credit'] = df_loans['bc_open_to_buy'] + (df_loans['total_bc_limit'] - df_loans['total_bal_ex_mort'])

# Credit utilization overall
df_loans['total_util'] = df_loans['total_bal_ex_mort'] / df_loans['tot_hi_cred_lim']

# Accounts per year of credit history
df_loans['accts_per_year'] = df_loans['total_acc'] / (df_loans['credit_history_months'] / 12)

# Delinquency rate 
df_loans['delinq_rate'] = df_loans['delinq_2yrs'] / df_loans['total_acc'].replace(0, 1)

df_loans['available_credit'].hist()

# ENCODING: no, for now
'''
# Categorical Variables

cat_cols = ['addr_state', 'purpose', 'home_ownership', 'verification_status', 
            'emp_length', 'zip_code']

# Encoding
from sklearn.preprocessing import LabelEncoder
for col in cat_cols:
    le = LabelEncoder()
    df_loans[col + '_encoded'] = le.fit_transform(df_loans[col].astype(str))
    cols_ignore += [col]
'''

# ## 3. Exploratory

df_loans[df_loans['default'] == 1]['purpose'].value_counts()

# Cats
cat_cols = ['purpose', 'home_ownership', 'addr_state', 'emp_length']

for col in cat_cols:
    print(f"\n{col}:")
    print(df_loans.groupby(col)['default'].agg(['mean', 'count']).sort_values('mean', ascending = False).head(10))

bin_cols = ['income_bin', 'dti_bin', 'loan_amnt_bin', 'revol_util_bin']
all_cols = [col for col in df_loans.columns if (col not in cols_ignore) and (col not in cat_cols) and (col not in bin_cols)]

df_loans.groupby('default')[all_cols].median()

# Correlations

numeric_cols = df_loans.select_dtypes(include=[np.number]).columns.difference(cols_ignore)
correlations = df_loans[numeric_cols].corrwith(df_loans['default']).sort_values(ascending = False)
print(correlations.head(20))

# Correlation matrix
corr_matrix = df_loans[numeric_cols].corr()

# Find highly correlated pairs (>0.8 or 0.9)
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:  # threshold
            high_corr.append((corr_matrix.columns[i], 
                            corr_matrix.columns[j], 
                            corr_matrix.iloc[i, j]))

print("Highly correlated pairs (>0.8):")
for col1, col2, corr_val in high_corr:
    print(f"{col1} <-> {col2}: {corr_val:.3f}")

# Definitions are slightly worse, dropping when corr is higher than 0.9
cols_ignore += ['mo_sin_old_rev_tl_op', 'num_actv_rev_tl', 'num_sats', 'tot_hi_cred_lim']

import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')  # If this doesn't work, remove this line

# ==================== GRAPH 1: Default Rate Overview ====================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1.1: Overall default rate
default_rate = df_loans['default'].mean()
axes[0, 0].bar(['Fully Paid', 'Charged Off'], 
               [1-default_rate, default_rate], 
               color=['#2ecc71', '#e74c3c'], 
               alpha=0.8, 
               edgecolor='black',
               linewidth=2)
axes[0, 0].set_ylabel('Proportion', fontsize=12, fontweight='bold')
axes[0, 0].set_title(f'Overall Default Rate: {default_rate:.2%}', fontsize=14, fontweight='bold')
axes[0, 0].set_ylim(0, 1)
axes[0, 0].grid(alpha=0.3, axis='y')
for i, v in enumerate([1-default_rate, default_rate]):
    axes[0, 0].text(i, v + 0.02, f'{v:.1%}', ha='center', fontsize=12, fontweight='bold')

# 1.2: Default rate over time
time_default = df_loans.groupby(df_loans['issue_d'].dt.to_period('Q'))['default'].agg(['mean', 'count'])
time_default.index = time_default.index.to_timestamp()
axes[0, 1].plot(time_default.index, time_default['mean'] * 100, marker='o', linewidth=2.5, markersize=7, color='#e74c3c', markeredgecolor='black', markeredgewidth=1)
axes[0, 1].set_xlabel('Issue Date', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Default Rate (%)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Default Rate Trends Over Time', fontsize=14, fontweight='bold')
axes[0, 1].grid(alpha=0.3)
axes[0, 1].tick_params(axis='x', rotation=45)

# 1.3: Default rate by loan purpose
purpose_default = df_loans.groupby('purpose')['default'].agg(['mean', 'count']).sort_values('mean', ascending=False)
top_purposes = purpose_default.head(10)
colors_purpose = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(top_purposes)))
bars = axes[1, 0].barh(range(len(top_purposes)), top_purposes['mean'] * 100, color=colors_purpose, alpha=0.8, edgecolor='black', linewidth=1.5)
axes[1, 0].set_yticks(range(len(top_purposes)))
axes[1, 0].set_yticklabels(top_purposes.index, fontsize=10)
axes[1, 0].set_xlabel('Default Rate (%)', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Default Rate by Loan Purpose (Top 10)', fontsize=14, fontweight='bold')
axes[1, 0].grid(axis='x', alpha=0.3)
for i, (idx, row) in enumerate(top_purposes.iterrows()):
    axes[1, 0].text(row['mean'] * 100 + 0.5, i, f"{row['mean']:.1%}\n(n={row['count']})", va='center', fontsize=8)

# 1.4: Default rate by home ownership
home_default = df_loans.groupby('home_ownership')['default'].agg(['mean', 'count']).sort_values('mean', ascending=False)
colors_home = ['#e74c3c' if x > default_rate else '#2ecc71' for x in home_default['mean']]
bars = axes[1, 1].bar(range(len(home_default)), home_default['mean'] * 100, color=colors_home, alpha=0.8, edgecolor='black', linewidth=1.5)
axes[1, 1].set_xticks(range(len(home_default)))
axes[1, 1].set_xticklabels(home_default.index, rotation=45, ha='right', fontsize=10)
axes[1, 1].set_ylabel('Default Rate (%)', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Default Rate by Home Ownership', fontsize=14, fontweight='bold')
axes[1, 1].axhline(y=default_rate * 100, color='black', linestyle='--', linewidth=2, label=f'Avg: {default_rate:.1%}')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(axis='y', alpha=0.3)
for i, (idx, row) in enumerate(home_default.iterrows()):
    axes[1, 1].text(i, row['mean'] * 100 + 1, f"{row['mean']:.1%}\n(n={row['count']})", ha='center', fontsize=8)

plt.tight_layout()
plt.show()

# ==================== GRAPH 2: Key Numeric Features ====================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

key_vars = ['score', 'annual_inc', 'dti', 'revol_util', 'delinq_2yrs', 'credit_history_months']

for idx, var in enumerate(key_vars):
    row, col = idx // 3, idx % 3
    
    defaulters = df_loans[df_loans['default'] == 1][var].dropna()
    non_defaulters = df_loans[df_loans['default'] == 0][var].dropna()
    
    bp = axes[row, col].boxplot([non_defaulters, defaulters], 
                                  labels=['Fully Paid', 'Charged Off'],
                                  patch_artist=True,
                                  showfliers=False,
                                  widths=0.6)
    
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_alpha(0.7)
    bp['boxes'][0].set_edgecolor('black')
    bp['boxes'][1].set_edgecolor('black')
    bp['boxes'][0].set_linewidth(1.5)
    bp['boxes'][1].set_linewidth(1.5)
    
    for whisker in bp['whiskers']:
        whisker.set(linewidth=1.5)
    for cap in bp['caps']:
        cap.set(linewidth=1.5)
    for median in bp['medians']:
        median.set(color='darkblue', linewidth=2)
    
    axes[row, col].set_ylabel(var, fontsize=11, fontweight='bold')
    axes[row, col].set_title(f'{var} by Default Status', fontsize=12, fontweight='bold')
    axes[row, col].grid(axis='y', alpha=0.3)
    
    med_non_def = non_defaulters.median()
    med_def = defaulters.median()
    axes[row, col].text(0.5, 0.95, f'Median:\nPaid: {med_non_def:.0f}\nDefault: {med_def:.0f}', 
                        transform=axes[row, col].transAxes, fontsize=9, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# ==================== GRAPH 3: Binned Variables Impact ====================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

bin_cols_to_plot = ['income_bin', 'dti_bin', 'loan_amnt_bin', 'revol_util_bin']

for idx, col in enumerate(bin_cols_to_plot):
    row, col_idx = idx // 2, idx % 2
    
    bin_stats = df_loans.groupby(col)['default'].agg(['mean', 'count']).reset_index()
    bin_stats = bin_stats.sort_values('mean', ascending=False)
    
    colors_bin = plt.cm.plasma(np.linspace(0.2, 0.8, len(bin_stats)))
    bars = axes[row, col_idx].barh(range(len(bin_stats)), bin_stats['mean'] * 100, 
                                     color=colors_bin, alpha=0.85, edgecolor='black', linewidth=1.5)
    axes[row, col_idx].set_yticks(range(len(bin_stats)))
    axes[row, col_idx].set_yticklabels(bin_stats[col], fontsize=10)
    axes[row, col_idx].set_xlabel('Default Rate (%)', fontsize=12, fontweight='bold')
    axes[row, col_idx].set_title(f'Default Rate by {col}', fontsize=13, fontweight='bold')
    axes[row, col_idx].grid(axis='x', alpha=0.3)
    
    for i, (_, row_data) in enumerate(bin_stats.iterrows()):
        axes[row, col_idx].text(row_data['mean'] * 100 + 0.5, i, 
                                 f"{row_data['mean']:.1%} (n={int(row_data['count'])})", 
                                 va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

# ==================== GRAPH 4: Correlation Heatmap (Manual) ====================
plt.figure(figsize=(14, 11))

corr_vars = ['default', 'score', 'annual_inc', 'dti', 'revol_util', 'delinq_2yrs', 
             'pub_rec', 'inq_last_6mths', 'open_acc', 'total_acc', 'credit_history_months',
             'num_actv_bc_tl', 'bc_util', 'tot_cur_bal', 'mo_sin_old_rev_tl_op']
corr_vars = [v for v in corr_vars if v in df_loans.columns]

corr_matrix = df_loans[corr_vars].corr()

im = plt.imshow(corr_matrix, cmap='RdYlGn_r', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(im, shrink=0.8)

plt.xticks(range(len(corr_vars)), corr_vars, rotation=45, ha='right', fontsize=10)
plt.yticks(range(len(corr_vars)), corr_vars, fontsize=10)

for i in range(len(corr_vars)):
    for j in range(len(corr_vars)):
        text = plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)

plt.title('Correlation Matrix: Key Features vs Default', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# ==================== GRAPH 5: Feature Importance Preview ====================
plt.figure(figsize=(12, 8))

numeric_features = [col for col in df_loans.select_dtypes(include=[np.number]).columns 
                    if col not in cols_ignore and col != 'default']
correlations = df_loans[numeric_features + ['default']].corr()['default'].drop('default').sort_values(key=abs, ascending=False)

top_corr = correlations.head(20)
colors_corr = ['#e74c3c' if x > 0 else '#2ecc71' for x in top_corr.values]

plt.barh(range(len(top_corr)), top_corr.values, color=colors_corr, alpha=0.85, edgecolor='black', linewidth=1.5)
plt.yticks(range(len(top_corr)), top_corr.index, fontsize=10)
plt.xlabel('Correlation with Default', fontsize=12, fontweight='bold')
plt.title('Top 20 Features Correlated with Default (by Absolute Value)', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
plt.grid(axis='x', alpha=0.3)

for i, (idx, val) in enumerate(top_corr.items()):
    plt.text(val + 0.005 if val > 0 else val - 0.005, i, f'{val:.3f}', 
             va='center', ha='left' if val > 0 else 'right', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.show()

# ==================== Summary Statistics ====================
print("=" * 80)
print("KEY INSIGHTS TO HIGHLIGHT IN YOUR PRESENTATION:")
print("=" * 80)
print(f"1. Overall default rate: {default_rate:.2%}")
print(f"2. Total loans analyzed: {len(df_loans):,}")
print(f"3. Highest risk loan purposes: {', '.join(purpose_default.head(3).index.tolist())}")
print(f"4. Highest risk home ownership: {home_default.index[0]}")
print(f"5. Median score - Defaulters: {df_loans[df_loans['default']==1]['score'].median():.1f} vs Non-defaulters: {df_loans[df_loans['default']==0]['score'].median():.1f}")
print(f"6. Top predictors (by correlation):")
for i, (feat, corr_val) in enumerate(correlations.head(5).items(), 1):
    print(f"   {i}. {feat}: {corr_val:.3f}")
print("=" * 80)



