
# Cell 1: Imports and setup
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score, precision_score,
                             recall_score, confusion_matrix, classification_report)
import joblib
import statsmodels.api as sm

os.makedirs('csv_files', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

plt.rcParams['figure.figsize'] = (10,5)

# %%
# Cell 2: Download datasets
candidate_name = 'Narayana_Reddy'

trade_data_gdrive = 'https://drive.google.com/uc?id=1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs&export=download'
fear_greed_gdrive = 'https://drive.google.com/uc?id=1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf&export=download'

trade_csv = 'historical_trades.csv'
fear_csv = 'fear_greed.csv'

import gdown

print('Downloading trade dataset...')
try:
    gdown.download(trade_data_gdrive, trade_csv, quiet=False)
except Exception as e:
    print('Download failed')

print('Downloading Fear & Greed dataset...')
try:
    gdown.download(fear_greed_gdrive, fear_csv, quiet=False)
except Exception as e:
    print('Download failed')

# %%
# Cell 3: Load datasets
trade_df = pd.read_csv(trade_csv, low_memory=False)
fear_df = pd.read_csv(fear_csv, low_memory=False)

print('Trade data shape:', trade_df.shape)
print(trade_df.head())

print('\nFear/Greed data shape:', fear_df.shape)
print(fear_df.head())

# %%
# Cell 4: Clean and parse dates
for col in ['time','timestamp','created_at','datetime']:
    if col in trade_df.columns:
        trade_df['time'] = pd.to_datetime(trade_df[col], errors='coerce')
        break

if 'time' not in trade_df.columns or trade_df['time'].isnull().all():
    cand = [c for c in trade_df.columns if 'time' in c.lower() or 'date' in c.lower()]
    if cand:
        trade_df['time'] = pd.to_datetime(trade_df[cand[0]], errors='coerce')

date_candidates = [c for c in fear_df.columns if 'date' in c.lower()]
if date_candidates:
    fear_df['Date'] = pd.to_datetime(fear_df[date_candidates[0]], errors='coerce')
else:
    fear_df['Date'] = pd.to_datetime(fear_df.iloc[:,0], errors='coerce')

sent_candidates = [c for c in fear_df.columns if 'class' in c.lower() or 'fear' in c.lower() or 'greed' in c.lower()]
if sent_candidates:
    fear_df['Sentiment'] = fear_df[sent_candidates[0]].astype(str)
else:
    fear_df['Sentiment'] = fear_df.iloc[:,1].astype(str)

fear_df = fear_df.dropna(subset=['Date'])

print('\nTrade time nulls:', trade_df['time'].isnull().sum())
print('Fear data preview:')
print(fear_df[['Date','Sentiment']].head())

# %%
# Cell 5: Standardize columns
trade_df.columns = [c.strip() for c in trade_df.columns]

col_map = {c.lower():c for c in trade_df.columns}

def get_col(names):
    for n in names:
        if n in col_map:
            return col_map[n]
    return None

exec_col = get_col(['execution price','execution_price','price','exec_price'])
size_col = get_col(['size','quantity','qty'])
pnl_col = get_col(['closedPnL','closedpnl','pnl','profit'])
lev_col = get_col(['leverage','lev'])
side_col = get_col(['side'])
account_col = get_col(['account','acct','user'])
symbol_col = get_col(['symbol','pair'])

print('Columns detected ->', exec_col, size_col, pnl_col, lev_col, side_col)

trade_df['exec_price'] = pd.to_numeric(trade_df[exec_col], errors='coerce') if exec_col else np.nan
trade_df['size'] = pd.to_numeric(trade_df[size_col], errors='coerce') if size_col else np.nan
trade_df['pnl'] = pd.to_numeric(trade_df[pnl_col], errors='coerce') if pnl_col else np.nan
trade_df['leverage'] = pd.to_numeric(trade_df[lev_col], errors='coerce') if lev_col else np.nan
trade_df['side'] = trade_df[side_col] if side_col else np.nan
trade_df['account'] = trade_df[account_col] if account_col else np.nan
trade_df['symbol'] = trade_df[symbol_col] if symbol_col else np.nan

trade_df['date'] = pd.to_datetime(trade_df['time']).dt.date

trade_df.to_csv('csv_files/trades_cleaned.csv', index=False)
print('Cleaned trade data saved')

# %%
# Cell 6: Aggregate metrics
agg = trade_df.groupby('date').agg(
    total_trades = ('pnl','count'),
    total_volume = ('size', lambda x: np.nansum(np.abs(x))),
    avg_leverage = ('leverage', 'mean'),
    win_rate = ('pnl', lambda x: np.nanmean((x>0).astype(int))),
    avg_pnl = ('pnl', 'mean'),
    median_pnl = ('pnl', 'median'),
    std_pnl = ('pnl', 'std')
).reset_index()

agg['date'] = pd.to_datetime(agg['date'])
agg.to_csv('csv_files/daily_trade_metrics.csv', index=False)

fear_daily = fear_df[['Date','Sentiment']].copy()
fear_daily['Date'] = pd.to_datetime(fear_daily['Date']).dt.date
fear_daily = fear_daily.rename(columns={'Date':'date'})
fear_daily['date'] = pd.to_datetime(fear_daily['date'])

merged = pd.merge(agg, fear_daily, on='date', how='left')
merged.to_csv('csv_files/daily_merged.csv', index=False)

# %%
# Cell 7: Visualizations
plt.figure(figsize=(8,5))
sns.boxplot(data=merged.dropna(subset=['Sentiment']), x='Sentiment', y='avg_pnl')
plt.title('Distribution of average daily PnL by Market Sentiment')
plt.savefig('outputs/boxplot_avg_pnl_by_sentiment.png')
plt.show()

plt.figure(figsize=(14,5))
plt.plot(merged['date'], merged['avg_pnl'], label='avg_pnl')
for idx, row in merged.dropna(subset=['Sentiment']).iterrows():
    if row['Sentiment'].strip().lower().startswith('fear'):
        plt.axvspan(row['date'] - pd.Timedelta(days=0.5), row['date'] + pd.Timedelta(days=0.5), color='red', alpha=0.03)
    elif row['Sentiment'].strip().lower().startswith('greed'):
        plt.axvspan(row['date'] - pd.Timedelta(days=0.5), row['date'] + pd.Timedelta(days=0.5), color='green', alpha=0.03)
plt.title('Average daily PnL with Sentiment')
plt.savefig('outputs/time_series_avg_pnl_sentiment.png')
plt.show()

# %%
# Cell 8: Statistical testing
from scipy import stats

df_fg = merged.dropna(subset=['Sentiment'])
fear_vals = df_fg[df_fg['Sentiment'].str.contains('fear', case=False, na=False)]['avg_pnl'].dropna()
greed_vals = df_fg[df_fg['Sentiment'].str.contains('greed', case=False, na=False)]['avg_pnl'].dropna()

print('Fear count:', len(fear_vals), 'Greed count:', len(greed_vals))

if len(fear_vals)>1 and len(greed_vals)>1:
    tstat, pval = stats.ttest_ind(fear_vals, greed_vals, equal_var=False, nan_policy='omit')
    print('T-test result: t=%.4f, p=%.6f' % (tstat, pval))

def cohen_d(x,y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled = ( (nx-1)*np.nanvar(x, ddof=1) + (ny-1)*np.nanvar(y, ddof=1) ) / dof
    return (np.nanmean(x) - np.nanmean(y)) / np.sqrt(pooled)

if len(fear_vals)>1 and len(greed_vals)>1:
    print('Cohen d:', cohen_d(fear_vals, greed_vals))

# %%
# Cell 9: Correlation heatmap
numcols = ['total_trades','total_volume','avg_leverage','win_rate','avg_pnl','median_pnl','std_pnl']
plt.figure(figsize=(8,6))
cm = merged[numcols].corr()
sns.heatmap(cm, annot=True, fmt='.2f')
plt.title('Correlation matrix')
plt.savefig('outputs/correlation_heatmap.png')
plt.show()

# %%
# Cell 10: Trade-level modeling
trade_df['date'] = pd.to_datetime(trade_df['date'])
sent_map = fear_daily.set_index('date')['Sentiment'].to_dict()
trade_df['Sentiment'] = trade_df['date'].map(sent_map)

trade_df['label_profit'] = (trade_df['pnl']>0).astype(int)
trade_df['abs_size'] = np.abs(trade_df['size'])
trade_df['side_buy'] = (trade_df['side'].astype(str).str.lower()=='buy').astype(int)

def encode_sent(s):
    if pd.isna(s):
        return np.nan
    s = str(s).lower()
    if 'fear' in s:
        return 1
    if 'greed' in s:
        return 0
    return np.nan

trade_df['sentiment_code'] = trade_df['Sentiment'].apply(encode_sent)

model_df = trade_df[['label_profit','abs_size','leverage','exec_price','side_buy','sentiment_code']].copy()
model_df = model_df.dropna()

X = model_df.drop(columns=['label_profit'])
y = model_df['label_profit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

importances = clf.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print('Feature importances:\n', feat_imp)
feat_imp.plot(kind='bar')
plt.title('Feature importances')
plt.savefig('outputs/feature_importances.png')
plt.show()

joblib.dump(clf, f'csv_files/{candidate_name}_rf_model.joblib')
joblib.dump(scaler, f'csv_files/{candidate_name}_scaler.joblib')

# %%
# Cell 11: Save summary
summary = []
summary.append('Candidate: %s' % candidate_name)
summary.append('Date: %s' % datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'))
summary.append('- Data download and cleaning')
summary.append('- Daily aggregation of trading metrics')
summary.append('- Merge with Fear/Greed sentiment')
summary.append('- EDA and visualizations')
summary.append('- Statistical testing')
summary.append('- Trade-level model with sentiment feature')

with open('csv_files/summary.txt','w') as f:
    f.write('\n'.join(summary))

print('\n'.join(summary))

# %%
# Cell 12: Package submission
import shutil
shutil.make_archive('submission_outputs', 'zip', root_dir='.')
print('submission_outputs.zip created')
