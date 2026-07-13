import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Veriyi Yükleme
df = pd.read_csv("employment_08_09.csv")

# 2. Veri Ön İşleme
df = df.dropna(subset=['earnwke'])
X = df.drop(columns=['earnwke'])
y = df['earnwke']

# Eğitim ve Test Ayrımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Eğitimi (Gradient Boosting)
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Değerlendirme
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- Gradyan Artırma (Gradient Boosting Regressor) ---")
print(f"Ortalama Kare Hata (MSE): {mse:.2f}")
print(f"R-Kare (R²) Skoru     : {r2:.4f}")
