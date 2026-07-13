FROM python:3.11-slim

WORKDIR /app

# Önce requirements kopyalanır (Cache avantajı için bu yaptığın çok doğru bir hamle)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tüm Python scriptlerini, CSV veri setini ve diğer gereksinimleri içeri kopyalar
COPY . .

# Eğer uygulaman web sunucusu (Flask/FastAPI vb.) değilse EXPOSE'a teknik olarak gerek yoktur, 
# ancak bilgi amaçlı kalabilir.
EXPOSE 3000

# Docker Compose dosyasındaki "command: python ..." satırları bu imajı her servis için 
# farklı bir görevle başlatacağı için, buraya varsayılan bir bekleme komutu koymak yeterlidir.
CMD ["python", "--version"]ß    