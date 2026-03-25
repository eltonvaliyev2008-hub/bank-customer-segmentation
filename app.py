from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

app = FastAPI(title="Bank Müştəri Seqmentasiyası API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

model  = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

cluster_adlari = {
    0: 'Gənc/Digital Müştəri',
    1: 'Çox Riskli Müştəri',
    2: 'Premium Müştəri',
    3: 'Orta Müştəri',
    4: 'Riskli Müştəri'
}

cluster_tavsiye = {
    0: 'Mobil tətbiq kampaniyaları təklif edin. Rəqəmsal xidmətlərə fokuslanın.',
    1: 'Kredit limitini azaldın. Gecikmiş ödənişlər üçün xəbərdarlıq sistemi qurun.',
    2: 'VIP xidmət paketləri, investisiya məhsulları və eksklüziv faiz dərəcələri təklif edin.',
    3: 'Kredit məhsulları və əmanət faizlərini artırma kampaniyaları aparın.',
    4: 'Maliyyə məsləhəti təklif edin. Borc restrukturizasiyasını nəzərdən keçirin.'
}

def musteri_dataframe_yarat(musteri):
    return pd.DataFrame([{
        'yaş'                    : musteri.yas,
        'şəhər'                  : musteri.seher,
        'peşə'                   : musteri.pese,
        'təhsil'                 : musteri.tehsil,
        'ailə_vəziyyəti'         : musteri.aile_veziyyeti,
        'maaş_azn'               : musteri.maas_azn,
        'balans_azn'             : musteri.balans_azn,
        'depozit_azn'            : musteri.depozit_azn,
        'kredit_balı'            : musteri.kredit_bali,
        'kredit_borcu_azn'       : musteri.kredit_borcu_azn,
        'borc_gəlir_nisbəti'     : musteri.borc_gelir_nisbeti,
        'bank_stajı_il'          : musteri.bank_staji_il,
        'aylıq_əməliyyat_sayı'   : musteri.aylik_emeliyyat_sayi,
        'aylıq_xərc_azn'         : musteri.aylik_xerc_azn,
        'onlayn_əməliyyat_faizi' : musteri.onlayn_emeliyyat_faizi,
        'gecikmiş_ödəniş_sayı'   : musteri.gecikmis_odenis_sayi,
        'kart_sayı'              : musteri.kart_sayi,
        'sığorta_sayı'           : musteri.sigorta_sayi,
        'bank_məhsulları'        : musteri.bank_mehsullari,
        'churn_riski'            : musteri.churn_riski
    }])

class Musteri(BaseModel):
    yas                    : float
    seher                  : int
    pese                   : int
    tehsil                 : int
    aile_veziyyeti         : int
    maas_azn               : float
    balans_azn             : float
    depozit_azn            : float
    kredit_bali            : int
    kredit_borcu_azn       : float
    borc_gelir_nisbeti     : float
    bank_staji_il          : int
    aylik_emeliyyat_sayi   : int
    aylik_xerc_azn         : float
    onlayn_emeliyyat_faizi : float
    gecikmis_odenis_sayi   : int
    kart_sayi              : int
    sigorta_sayi           : int
    bank_mehsullari        : int
    churn_riski            : float

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    return templates.TemplateResponse(request=request, name="admin.html")

@app.get("/health")
def health():
    return {"status": "API işləyir", "version": "1.0"}

@app.post("/proqnoz")
def proqnoz(musteri: Musteri):
    data    = musteri_dataframe_yarat(musteri)
    scaled  = scaler.transform(data)
    netice  = model.predict(scaled)[0]
    ehtimal = model.predict_proba(scaled)[0]
    return {
        "cluster" : int(netice),
        "netice"  : cluster_adlari[netice],
        "ehtimal" : round(float(max(ehtimal)) * 100, 2)
    }

@app.post("/admin/proqnoz")
def admin_proqnoz(musteri: Musteri):
    data    = musteri_dataframe_yarat(musteri)
    scaled  = scaler.transform(data)
    netice  = model.predict(scaled)[0]
    ehtimal = model.predict_proba(scaled)[0]
    return {
        "cluster" : int(netice),
        "netice"  : cluster_adlari[netice],
        "ehtimal" : round(float(max(ehtimal)) * 100, 2),
        "tavsiye" : cluster_tavsiye[netice]
    }
