from fastapi import FastAPI, Request
import httpx
import os

app = FastAPI()

# Arka plandaki Docker konteynerlerinin isimleri ve portları
SERVICES = {
    "sentiment": os.getenv("SENTIMENT_URL", "http://ml-sentiment:5001"),
    "churn": os.getenv("CHURN_URL", "http://ml-churn:5002"),
    "recommend": os.getenv("RECOMMEND_URL", "http://ml-recommend:5003"),
    "forecast": os.getenv("FORECAST_URL", "http://ml-forecast:5004"),
    "segmentation": os.getenv("SEGMENTATION_URL", "http://ml-segmentation:5005"),
}

@app.get("/")
def home():
    return {"message": "API Gateway Calisiyor! Tum servislere buradan ulasabilirsiniz."}

@app.post("/api/{service_name}/predict")
async def route_request(service_name: str, request: Request):
    # İstenen servis listemizde var mı kontrol et
    if service_name not in SERVICES:
        return {"error": f"'{service_name}' adinda bir servis bulunamadi."}
    
    # İsteği ilgili servise yönlendir
    target_url = f"{SERVICES[service_name]}/predict"
    try:
        payload = await request.json()
        async with httpx.AsyncClient() as client:
            # Gateway isteği alır, ML servisine iletir ve cevabı bekler
            resp = await client.post(target_url, json=payload, timeout=10.0)
        return resp.json()
    except Exception as e:
        return {"error": f"{service_name} servisine ulasilamadi veya servis uykuda.", "details": str(e)}
