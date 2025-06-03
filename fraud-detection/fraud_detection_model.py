import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Load your standard scalre
with open('fraud_model_artifacts/xgb_model.pkl', 'rb') as file:
    xgboost_model = pickle.load(file)

# Load your pre-trained model
with open('fraud_model_artifacts/standard_scaler.pkl', 'rb') as file:
    sc_model = pickle.load(file)

model_threshold = 0.7

def get_histoical_trans_df(from_acc_id):
  # sql code to connect and get data
  return pd.DataFrame()

def generate_agg_feat_df(from_acc_id):
  hist_df = get_histoical_trans_df(from_acc_id)
  if not hist_df.empty:
    fraud_ratio = hist_df.groupby('from_acc_id')['is_fradulent'].mean().rename('fraud_ratio').iloc[0]
    total_transactions_per_account = hist_df.groupby('from_acc_id').size().rename('total_transactions').iloc[0]
    average_amount_per_account = hist_df.groupby('from_acc_id')['transaction_amount'].mean().rename('avg_amount_per_account').iloc[0]
  else:
    fraud_ratio = 0
    total_transactions_per_account = 0
    average_amount_per_account = 0

  return fraud_ratio, total_transactions_per_account, average_amount_per_account

def generate_features(df):

  feat_df = df.copy()

  # 0. get aggregation for from_acc_id
  fraud_ratio, total_transactions_per_account, average_amount_per_account = generate_agg_feat_df(feat_df['from_acc_id'].iloc[0])

  # 1. Transaction Amount Logarithm
  feat_df['log_transaction_amount'] = np.log(feat_df['transaction_amount'] + 1)

  # 2. Transaction Date Features (Year, Month, Day of Week, Hour)
  feat_df['trans_date'] = pd.to_datetime(feat_df['trans_date'])
  feat_df['trans_year'] = feat_df['trans_date'].dt.year
  feat_df['trans_month'] = feat_df['trans_date'].dt.month
  feat_df['trans_day'] = feat_df['trans_date'].dt.day
  feat_df['trans_day_of_week'] = feat_df['trans_date'].dt.dayofweek
  feat_df['trans_hour'] = feat_df['trans_date'].dt.hour

  # 3. Interaction Features
  feat_df['amount_status_interaction'] = feat_df['transaction_amount'] * feat_df['status_type_id']

  # 4. Aggregated Features
  # Total Transactions and Average Amount for each 'from_acc_id'
  feat_df["total_transactions"] = total_transactions_per_account
  feat_df["avg_amount_per_account"] = average_amount_per_account

  # 5. Encode Categorical Features
  # Simple One Hot Encoding for transactional Type and Status
  # feat_df = pd.get_dummies(feat_df, columns=['trans_type_ID', 'status_type_id'], prefix=['type', 'status'], drop_first=True)

  # 6. Fraud Ratio of each from_acc_id
  feat_df["fraud_ratio"] = fraud_ratio

  return feat_df

def predict_fraud(trans_data):

  skip_cols = ['transaction_id', 'transaction_amount', 'trans_date', 'from_acc_id', 'to_acc_id',]

  # converting dictionary to dataframe
  trans_df = pd.DataFrame(trans_data, index=[0])

  # generate features
  feat_df = generate_features(trans_df)
  feat_df.drop(columns=skip_cols, axis=1, inplace=True)

  # scale features
  # feature_cols = list(feat_df.columns)
  scaled_feat_df = sc_model.transform(feat_df)

  # Transform features to suitable input format
  # dmatrix = xgb.DMatrix(np.float32(scaled_feat_df), feature_names=feature_cols)

  # Predict using the model
  prediction = xgboost_model.predict(scaled_feat_df)

  # Process prediction for response
  is_fraudulent = (prediction > model_threshold).astype(int)

  return is_fraudulent