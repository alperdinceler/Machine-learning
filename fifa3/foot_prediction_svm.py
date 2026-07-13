import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import os

def main():
    print("--- FIFA Tercih Edilen Ayak Tahmini (SVM) Başlatılıyor ---\n")
    
    # 1. VERİ YÜKLEME VE KONTROL
    try:
        df = pd.read_csv("fifa_world_cup_2026_player_performance-selected-columns.csv")
        print("[+] Veri seti başarıyla yüklendi.")
    except FileNotFoundError:
        print("[-] HATA: CSV dosyası bulunamadı!")
        return

    # 2. VERİ ÖN İŞLEME
    # Modele Yaş, Boy ve Kilo bilgilerini verip Hangi Ayağını (Sol/Sağ) kullandığını bulmasını isteyeceğiz
    features = ['age', 'height_cm', 'weight_kg']
    target = 'preferred_foot'

    # Eksik verileri temizle
    ml_data = df.dropna(subset=features + [target])

    X = ml_data[features]
    y = ml_data[target]

    # SVM, verilerin ölçeklendirilmesine (scaling) çok duyarlıdır. Bu yüzden StandardScaler KULLANMAK ZORUNDAYIZ.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Veriyi Eğitim (%80) ve Test (%20) olarak ayır
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 3. MODEL EĞİTİMİ (SVM)
    print("[+] Destek Vektör Makinesi (SVM) algoritması eğitiliyor...")
    # kernel='rbf' (Radial Basis Function) genelde en iyi sonuç veren çekirdektir.
    svm_model = SVC(kernel='rbf', random_state=42) 
    svm_model.fit(X_train, y_train)

    # 4. MODEL DEĞERLENDİRMESİ
    print("[+] Test verileri üzerinde ayak tercihleri tahmin ediliyor...")
    y_pred = svm_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"[+] Model Genel Doğruluk Oranı: %{acc * 100:.2f}\n")

    # 5. SONUÇLARIN RAPORLANMASI
    output_dir = "/app/output"
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, "foot_prediction_svm_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("--- SVM AYAK TERCİHİ TAHMİN MODELİ SONUÇLARI ---\n\n")
        f.write(f"Genel Doğruluk Oranı: %{acc * 100:.2f}\n\n")
        f.write("Detaylı Sınıflandırma Raporu:\n")
        f.write(classification_report(y_test, y_pred, zero_division=0))
        
    print(f"--- İŞLEM BİTTİ ---")
    print(f"Detaylı analiz raporu '{report_path}' konumuna kaydedildi.")

if __name__ == "__main__":
    main()