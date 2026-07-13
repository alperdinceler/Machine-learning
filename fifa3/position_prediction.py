import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import os

def main():
    print("--- FIFA Mevki (Pozisyon) Tahmin Modeli Başlatılıyor ---\n")
    
    # 1. VERİ YÜKLEME VE KONTROL
    try:
        df = pd.read_csv("fifa_world_cup_2026_player_performance-selected-columns.csv")
        print("[+] Veri seti başarıyla yüklendi.")
    except FileNotFoundError:
        print("[-] HATA: CSV dosyası bulunamadı!")
        return

    # 2. VERİ ÖN İŞLEME
    # Modele sadece Boy ve Kilo bilgilerini verip mevkisini bulmasını isteyeceğiz
    features = ['height_cm', 'weight_kg']
    target = 'position'

    # Modelin kafasını karıştırmamak için eksik verileri temizle
    ml_data = df.dropna(subset=features + [target])

    X = ml_data[features]
    y = ml_data[target]

    # Veriyi Eğitim (%80) ve Test (%20) olarak ikiye böl
    # Model %80'lik kısımla öğrenecek, hiç görmediği %20'lik kısımla sınav olacak
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. MODEL EĞİTİMİ
    print("[+] Random Forest (Rastgele Orman) algoritması eğitiliyor...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 4. MODEL DEĞERLENDİRMESİ
    print("[+] Test verileri üzerinde mevkiler tahmin ediliyor...")
    y_pred = clf.predict(X_test)
    
    # Gerçek mevkilerle modelin tahminlerini karşılaştır
    acc = accuracy_score(y_test, y_pred)
    print(f"[+] Model Genel Doğruluk Oranı: %{acc * 100:.2f}\n")

    # 5. SONUÇLARIN RAPORLANMASI
    output_dir = "/app/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Hangi mevkileri daha iyi tahmin ettiğini gösteren detaylı rapor
    report_path = os.path.join(output_dir, "position_prediction_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("--- MEVKİ TAHMİN MODELİ SONUÇLARI ---\n\n")
        f.write(f"Genel Doğruluk Oranı: %{acc * 100:.2f}\n\n")
        f.write("Detaylı Sınıflandırma Raporu (Hangi mevki ne kadar doğru bilindi?):\n")
        f.write(classification_report(y_test, y_pred))
        
    print(f"--- İŞLEM BİTTİ ---")
    print(f"Detaylı analiz raporu '{report_path}' konumuna kaydedildi.")

if __name__ == "__main__":
    main()