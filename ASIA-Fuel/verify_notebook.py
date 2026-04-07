import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)

def verify_ml_performance():
    try:
        # Load datasets
        df_asia_detailed = pd.read_csv('asia_fuel_prices_detailed.csv')
        df_global_prices = pd.read_csv('global_fuel_prices.csv')
        df_tax_comparison = pd.read_csv('fuel_tax_comparison.csv')

        # Feature Engineering (as in notebook)
        ml_df = pd.merge(df_global_prices, df_asia_detailed[['country', 'avg_monthly_income_usd', 'ev_adoption_pct', 'oil_import_dependency_pct']], on='country', how='left')
        ml_df['log_income'] = np.log1p(ml_df['avg_monthly_income_usd'].fillna(ml_df['avg_monthly_income_usd'].median()))
        ml_df['is_asian'] = ml_df['is_asian'].astype(int)
        ml_df = pd.merge(ml_df, df_tax_comparison[['country', 'total_tax_usd_per_liter']], on='country', how='left')
        ml_df['total_tax_usd_per_liter'] = ml_df['total_tax_usd_per_liter'].fillna(0)

        features = ['log_income', 'is_asian', 'total_tax_usd_per_liter', 'avg_fuel_usd']
        X = ml_df[features].dropna()
        y = ml_df.loc[X.index, 'gasoline_usd_per_liter']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=SEED)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        print(f"VERIFICATION: XGBoost R2 Score = {r2:.4f}")
        if r2 > 0.85:
            print("VERIFICATION: SUCCESS - R2 exceeds Gold Medal requirement (>0.85)")
        else:
            print("VERIFICATION: WARNING - R2 is below requirement.")
            
    except Exception as e:
        print(f"VERIFICATION: ERROR - {str(e)}")

if __name__ == "__main__":
    verify_ml_performance()
