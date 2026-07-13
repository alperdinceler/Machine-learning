import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os

def main():
    print("--- FIFA Oyuncu Kümeleme Analizi Başlatılıyor ---\n")
    
    # 1. VERİ YÜKLEME VE KONTROL
    try:
        df = pd.read_csv("fifa_world_cup_2026_player_performance-selected-columns.csv")
        print(f"[+] Veri seti yüklendi: Toplam {df.shape[0]} oyuncu, {df.shape[1]} özellik bulundu.")
    except FileNotFoundError:
        print("[-] HATA: CSV dosyası bulunamadı! Lütfen dosya adını ve konumunu kontrol edin.")
        return

    # 2. VERİ ÖN İŞLEME (PREPROCESSING)
    # Modele sadece yaş, boy ve kilo özelliklerini veriyoruz.
    features = ['age', 'height_cm', 'weight_kg']
    
    # Eksik (NaN) veri barındıran oyuncuları modelin çökmemesi için analizden çıkarıyoruz.
    ml_data = df[features].dropna()
    
    # Ölçeklendirme (Scaling): Boy cm, kilo kg cinsinden olduğu için değer aralıkları çok farklıdır.
    # K-Means mesafe tabanlı çalıştığı için tüm verileri standartlaştırıp (ortalama 0, varyans 1) aynı teraziye alıyoruz.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(ml_data)

    # 3. MODEL EĞİTİMİ (TRAINING)
    n_clusters = 4
    print(f"\n[+] K-Means modeli {n_clusters} farklı oyuncu profili (küme) için eğitiliyor...")
    
    # K-Means algoritmasını tanımla ve eğit
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Oluşan etiketleri (0, 1, 2, 3) ana veri setine yeni bir kolon olarak ekle.
    df.loc[ml_data.index, 'player_profile_cluster'] = cluster_labels

    # 4. MODEL DEĞERLENDİRMESİ (EVALUATION)
    # Silhouette Skoru: Kümelerin birbirlerinden ne kadar net ayrıldığını gösterir (-1 ile +1 arası).
    # 1'e ne kadar yakınsa, model oyuncuları o kadar başarılı gruplamış demektir.
    sil_score = silhouette_score(X_scaled, cluster_labels)
    print(f"[+] Model Silhouette Skoru: {sil_score:.3f}")

    # 5. SONUÇLARIN RAPORLANMASI VE KAYDEDİLMESİ
    output_dir = "/app/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Çıktı A: Detaylı Oyuncu Listesi (CSV)
    output_csv = os.path.join(output_dir, "detailed_player_clusters.csv")
    result_columns = ['player_name', 'team', 'position', 'age', 'height_cm', 'weight_kg', 'player_profile_cluster']
    df[result_columns].to_csv(output_csv, index=False)
    
    # Çıktı B: Yöneticiler/Antrenörler İçin Küme Özeti (TXT)
    summary_txt = os.path.join(output_dir, "cluster_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("--- FIFA OYUNCU PROFİLİ (KÜME) ÖZET RAPORU ---\n\n")
        
        # Her kümenin yaş, boy ve kilo ortalamalarını alıyoruz
        cluster_centers = df.groupby('player_profile_cluster')[features].mean()
        
        for cluster_id in range(n_clusters):
            # O kümeye ait oyuncuları filtrele
            cluster_data = df[df['player_profile_cluster'] == cluster_id]
            
            # Bu grupta en çok hangi mevki (pozisyon) yer alıyor?
            dominant_position = cluster_data['position'].mode()[0]
            
            f.write(f"KÜME {cluster_id}:\n")
            f.write(f" * Oyuncu Sayısı: {len(cluster_data)}\n")
            f.write(f" * Yaş Ortalaması: {cluster_centers.loc[cluster_id, 'age']:.1f} yıl\n")
            f.write(f" * Boy Ortalaması: {cluster_centers.loc[cluster_id, 'height_cm']:.1f} cm\n")
            f.write(f" * Kilo Ortalaması: {cluster_centers.loc[cluster_id, 'weight_kg']:.1f} kg\n")
            f.write(f" * En Sık Görülen Mevki: {dominant_position}\n")
            f.write("-" * 40 + "\n")

    print(f"\n--- İŞLEM BİTTİ ---")
    print(f"1. Tüm oyuncuların küme eşleşmeleri '{output_csv}' konumuna kaydedildi.")
    print(f"2. Küme özet raporu '{summary_txt}' konumuna kaydedildi.")

if __name__ == "__main__":
    main()