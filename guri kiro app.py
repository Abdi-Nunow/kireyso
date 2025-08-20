# app.py — GuriKire AI (MVP) — Jigjiga Customized
# Ku shaqeeya Streamlit + SQLite
# Astaamo cusub (sida aad codsatay):
#  - Magaalo waa default **Jigjiga** (kana dhigisnay disabled)
#  - Qabale (1 ilaa 17) — **selectbox**
#  - Nooca guri — **selectbox** (Apartment, Hal qol, 2 qol, 5 qol)
#  - Biyo — **selectbox** (Biyo leeyahay / Biyo ma leh)
#  - Raadinta waxay leedahay shaandhayn isku mid ah
#  - Talo-bixin AI + Qiyaas Qiime wali way shaqaynayaan, waxayna ka mid noqdeen astaamaha cusub

import os
import io
import base64
import sqlite3
from contextlib import closing
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

DB_PATH = "gurikire.db"

# ---------- Kaabayaasha DB ----------

def init_db():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        c = conn.cursor()
        # Jadwalka asaasiga ah
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS listings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                city TEXT,
                district TEXT,
                price REAL,
                bedrooms INTEGER,
                bathrooms INTEGER,
                area REAL,
                furnished INTEGER,
                created_at TEXT,
                image_blob BLOB
            )
            """
        )
        # Ku dar columns cusub haddii aysan jirin (SQLite: ADD COLUMN waa idempotent haddii aanan magaca ku celin)
        try:
            c.execute("ALTER TABLE listings ADD COLUMN qabal INTEGER")
        except sqlite3.OperationalError:
            pass
        try:
            c.execute("ALTER TABLE listings ADD COLUMN house_type TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            c.execute("ALTER TABLE listings ADD COLUMN water INTEGER")  # 1 = biyo leeyahay, 0 = ma leh
        except sqlite3.OperationalError:
            pass
        conn.commit()


def insert_listing(rec):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO listings
            (title, description, city, district, price, bedrooms, bathrooms, area, furnished, created_at, image_blob, qabal, house_type, water)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec.get("title"),
                rec.get("description"),
                rec.get("city"),
                rec.get("district"),
                rec.get("price"),
                rec.get("bedrooms"),
                rec.get("bathrooms"),
                rec.get("area"),
                1 if rec.get("furnished") else 0,
                datetime.utcnow().isoformat(),
                rec.get("image_blob"),
                rec.get("qabal"),
                rec.get("house_type"),
                1 if rec.get("water") else 0,
            ),
        )
        conn.commit()


def load_listings_df():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        df = pd.read_sql_query("SELECT * FROM listings ORDER BY created_at DESC", conn)
    if not df.empty:
        # Columns types
        for col in ["furnished", "bedrooms", "bathrooms"]:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)
        for col in ["price", "area"]:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(float)
        if "qabal" in df.columns:
            df["qabal"] = df["qabal"].fillna(0).astype(int)
        if "water" in df.columns:
            df["water"] = df["water"].fillna(0).astype(int)
    return df


# ---------- Caawiye: Sawirro ----------

def image_to_blob(uploaded_file):
    if uploaded_file is None:
        return None
    img = Image.open(uploaded_file).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def blob_to_image(blob):
    if blob is None:
        return None
    return Image.open(io.BytesIO(blob))


# ---------- AI: Recommendations ----------

def recommend_listings(df: pd.DataFrame, query_text: str, top_n: int = 5):
    if df.empty or not query_text.strip():
        return df.head(top_n)
    # samee text isku-dhaf ah si similarity u noqoto macno — ku dar house_type/qabal/water
    def water_txt(x):
        return "biyo_leh" if int(x) == 1 else "biyo_la'aan"
    parts = []
    if not df.empty:
        wt = df.get("water", pd.Series([0]*len(df))).apply(water_txt)
        parts = (
            df[["title", "description", "city", "district", "house_type"]]
            .fillna("")
            .agg(" ".join, axis=1)
            + " qabal " + df.get("qabal", pd.Series([0]*len(df))).astype(str)
            + " " + wt
        )
    corpus = parts.tolist() if len(parts) else []
    vectorizer = TfidfVectorizer(min_df=1, stop_words=None)
    mat = vectorizer.fit_transform(corpus + [query_text])
    sims = cosine_similarity(mat[-1], mat[:-1]).flatten() if len(corpus) else np.array([])
    df = df.copy()
    if len(sims):
        df["similarity"] = sims
        return df.sort_values("similarity", ascending=False).head(top_n)
    return df.head(top_n)


# ---------- AI: Price Estimator ----------

def train_price_model(df: pd.DataFrame):
    # Ku dar qabal, house_type, water
    usable = df.dropna(subset=["price"]) if not df.empty else df
    if usable is None or usable.empty or len(usable) < 4:
        return None
    feat_cols = ["city", "district", "qabal", "house_type", "bedrooms", "bathrooms", "area", "furnished", "water"]
    for col in feat_cols:
        if col not in usable.columns:
            usable[col] = 0
    X = usable[feat_cols]
    y = usable["price"]
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["city", "district", "house_type"]),
            ("num", "passthrough", ["qabal", "bedrooms", "bathrooms", "area", "furnished", "water"]),
        ]
    )
    model = Pipeline(steps=[("pre", pre), ("lr", LinearRegression())])
    model.fit(X, y)
    return model


def predict_price(model, sample: dict):
    if model is None:
        return None
    df_sample = pd.DataFrame([sample])
    try:
        pred = float(model.predict(df_sample)[0])
        return max(0.0, pred)
    except Exception:
        return None


# ---------- Chat Caawiye (placeholder) ----------

def helper_bot(message: str) -> str:
    msg = message.lower()
    hints = []
    if any(k in msg for k in ["qiime", "price", "kireyn", "kirada"]):
        hints.append("Waxaad isticmaalikartaa 'Qiyaas Qiime' si aad u aragto qiime suurtagal ah.")
    if any(k in msg for k in ["raadin", "raadso", "guri", "apartment", "degmo", "city", "qabal"]):
        hints.append("Tag 'Raadi Guri' oo isticmaal shaandhayn (qabal/nooc/biyo) si aad u hesho guri kugu habboon.")
    if any(k in msg for k in ["diiwaan", "ku dar", "listing"]):
        hints.append("Ku dar guri cusub adigoo buuxinaya foomka 'Ku dar Guri'.")
    if not hints:
        hints.append("Waa ku salaaman tahay! Iigu sheeg qabalka (1-17), nooca guriga, iyo biyo (leeyahay mise ma leh) — waan ku hagayaa.")
    return "\n\n".join(hints)


# ---------- UI ----------

st.set_page_config(page_title="GuriKire AI — Jigjiga", page_icon="🏠", layout="wide")
init_db()

QABALS = list(range(1, 18))  # 1 ilaa 17
HOUSE_TYPES = ["Apartment", "Hal qol", "2 qol", "5 qol"]
WATER_OPTS = ["Biyo leeyahay", "Biyo ma leh"]

st.sidebar.title("🏠 GuriKire AI — Jigjiga")
page = st.sidebar.radio("Dooro Bogga:", ["Raadi Guri", "Ku dar Guri", "Qiyaas Qiime", "Caawiye"], index=0)

st.sidebar.markdown("— **MVP**: xogtu waxa lagu kaydiyaa SQLite gudaha app-ka.")

if page == "Ku dar Guri":
    st.header("Ku dar Guri Kire ah — Jigjiga")
    with st.form("add_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            title = st.text_input("Cinwaan / Nooc (tusaale: 2-qol Apartment)")
            city = st.text_input("Magaalo", value="Jigjiga", disabled=True)
            qabal = st.selectbox("Qabale (1–17)", QABALS, index=0)
            house_type = st.selectbox("Nooca Guriga", HOUSE_TYPES, index=0)
            water_sel = st.selectbox("Biyo", WATER_OPTS, index=0)
            water_bool = True if water_sel == "Biyo leeyahay" else False
        with col2:
            district = st.text_input("Degmo / Xaafad (ikhtiyaar)")
            price = st.number_input("Qiimaha kirada (USD ama ETB)", min_value=0.0, step=10.0)
            area = st.number_input("Cabbirka (m²)", min_value=0.0, step=1.0)
            bedrooms = st.number_input("Qolal jiif", min_value=0, step=1)
            bathrooms = st.number_input("Musqulo", min_value=0, step=1)
            furnished = st.checkbox("Furnitured / Alaab yaal")
            image_file = st.file_uploader("Ku dar sawir (JPEG/PNG)", type=["jpg", "jpeg", "png"]) 
        description = st.text_area("Sharaxaad faahfaahsan")
        submitted = st.form_submit_button("💾 Kaydi Guri")
        if submitted:
            if not title:
                st.error("Fadlan geli ugu yaraan Cinwaan.")
            else:
                blob = image_to_blob(image_file)
                insert_listing({
                    "title": title,
                    "description": description,
                    "city": city,
                    "district": district,
                    "price": price,
                    "bedrooms": int(bedrooms),
                    "bathrooms": int(bathrooms),
                    "area": area,
                    "furnished": furnished,
                    "image_blob": blob,
                    "qabal": int(qabal),
                    "house_type": house_type,
                    "water": water_bool,
                })
                st.success("Waa la kaydiyay! 🟢")

elif page == "Raadi Guri":
    st.header("Raadi Guri — Jigjiga")
    df = load_listings_df()

    with st.expander("🔎 Shaandhayn"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            q_qabal = st.selectbox("Qabale", ["Dhammaan"] + QABALS, index=0)
        with c2:
            q_house = st.selectbox("Nooca Guriga", ["Dhammaan"] + HOUSE_TYPES, index=0)
        with c3:
            q_water = st.selectbox("Biyo", ["Dhammaan"] + WATER_OPTS, index=0)
        with c4:
            min_price, max_price = st.slider("Qiime (min, max)", 0, 100000, (0, 100000))
        q_text = st.text_input("Maxaad raadineysaa? (qoraal kooban)")

    df_f = df.copy()
    # City mar walba Jigjiga — lama shaandhaynayo magaalada
    if q_qabal != "Dhammaan" and "qabal" in df_f.columns:
        df_f = df_f[df_f["qabal"] == int(q_qabal)]
    if q_house != "Dhammaan" and "house_type" in df_f.columns:
        df_f = df_f[df_f["house_type"].str.lower() == q_house.lower()]
    if q_water != "Dhammaan" and "water" in df_f.columns:
        want = 1 if q_water == "Biyo leeyahay" else 0
        df_f = df_f[df_f["water"] == want]
    df_f = df_f[(df_f["price"] >= min_price) & (df_f["price"] <= max_price)]

    if q_text.strip():
        df_rec = recommend_listings(df_f, q_text, top_n=10)
    else:
        df_rec = df_f.head(10)

    if df_rec.empty:
        st.info("Weli xog yar ayaa ku jirta ama shaandhayntu waa adag. Isku day inaad ballaariso.")
    else:
        for _, row in df_rec.iterrows():
            with st.container(border=True):
                c1, c2 = st.columns([1, 2])
                with c1:
                    img = blob_to_image(row.get("image_blob")) if row.get("image_blob") is not None and not pd.isna(row.get("image_blob")) else None
                    if img is not None:
                        st.image(img, caption=row.get("title", ""), use_column_width=True)
                    else:
                        st.markdown("*(Ma jiro sawir la dhigay)*")
                with c2:
                    st.subheader(row.get("title", "")) 
                    qabal_txt = f"Qabal {int(row['qabal'])}" if not pd.isna(row.get("qabal")) else "Qabal n/a"
                    st.markdown(f"**Goob:** Jigjiga — {row.get('district','')}")
                    st.markdown(f"**Qabale:** {qabal_txt}")
                    st.markdown(f"**Nooc:** {row.get('house_type','')}")
                    st.markdown(f"**Biyo:** {'Leeyahay' if int(row.get('water',0))==1 else 'Ma leh'}")
                    st.markdown(f"**Qiime:** {row.get('price',0)}")
                    st.markdown(f"**Qolal/Musqulo:** {int(row.get('bedrooms',0))} / {int(row.get('bathrooms',0))}")
                    st.markdown(f"**Cabbir:** {row.get('area',0)} m² · **Alaab:** {'Haa' if int(row.get('furnished',0))==1 else 'Maya'}")
                    if "similarity" in row:
                        try:
                            st.progress(float(row["similarity"]))
                        except Exception:
                            pass
                    if row.get("description"):
                        st.write(row.get("description"))

elif page == "Qiyaas Qiime":
    st.header("Qiyaas qiimaha kirada — Jigjiga")
    df = load_listings_df()
    model = train_price_model(df) if not df.empty else None

    c1, c2, c3 = st.columns(3)
    with c1:
        city = st.text_input("Magaalo", value="Jigjiga", disabled=True)
        district = st.text_input("Degmo / Xaafad (ikhtiyaar)", value=df["district"].iloc[0] if not df.empty else "")
        qabal = st.selectbox("Qabale (1–17)", QABALS, index=0)
    with c2:
        house_type = st.selectbox("Nooca Guriga", HOUSE_TYPES, index=0)
        area = st.number_input("Cabbirka (m²)", min_value=0.0, step=1.0, value=float(df["area"].median()) if not df.empty else 80.0)
        water_sel = st.selectbox("Biyo", WATER_OPTS, index=0)
    with c3:
        bedrooms = st.number_input("Qolal jiif", min_value=0, step=1, value=int(df["bedrooms"].median()) if not df.empty else 2)
        bathrooms = st.number_input("Musqulo", min_value=0, step=1, value=int(df["bathrooms"].median()) if not df.empty else 1)
        furnished = st.checkbox("Furnitured / Alaab yaal", value=True if not df.empty and df["furnished"].mean() >= 0.5 else False)
        st.markdown("\n")
        run = st.button("🔮 Qiyaas Hadda")

    if run:
        sample = {
            "city": "Jigjiga",
            "district": district or "",
            "qabal": int(qabal),
            "house_type": house_type,
            "bedrooms": int(bedrooms),
            "bathrooms": int(bathrooms),
            "area": float(area),
            "furnished": 1 if furnished else 0,
            "water": 1 if water_sel == "Biyo leeyahay" else 0,
        }
        pred = predict_price(model, sample)
        if pred is None:
            st.warning("Xog kugu filan laguma hayo tababarka. Ku dar guryo badan si model-ku u barto.")
        else:
            st.success(f"Qiyaasta kirada: **{pred:,.0f}**")
            st.caption("*Tani waa qiyaas ku saleysan xogta la geliyay gudaha app-ka.*")

elif page == "Caawiye":
    st.header("Caawiye AI (MVP)")
    st.caption("Waxaad qori kartaa su'aal ku saabsan raadinta, qiimaha, ama sida loo isticmaalo app-ka.")
    user_msg = st.text_area("Su'aashaada / fariintaada")
    if st.button("U dir caawiye"):
        st.info(helper_bot(user_msg))

st.markdown("""
---
**Talooyin dejin:**
1. Ku socodsii gudaha: `pip install streamlit scikit-learn pillow pandas` kadibna `streamlit run app.py`.
2. Haddii aad rabto LLM dhab ah, u diyaari API KEY (tusaale `OPENAI_API_KEY`) oo ku dar wicitaan model gudaha `helper_bot`.
3. Si aad u dhoofiso, adeegso Streamlit Community Cloud ama Docker.
""")
