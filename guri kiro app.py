# app.py — GuriKire AI (MVP)
# Ku shaqeeya Streamlit + SQLite
# Astaamo:
#  - Diiwaangelinta guryaha (cinwaan, goob, qiime, qolal, sawir iwm)
#  - Raadinta & shaandhaynta
#  - Talo-bixin AI (recommendations) iyadoo la adeegsanayo TF‑IDF similarity
#  - Qiyaasta qiimaha (price estimator) iyadoo la adeegsanayo LinearRegression
#  - Chat-ka caawiyaha (placeholder) + meel lagu geliyo API haddii la doonayo
# Qoraalka UI-ga waa Af‑Soomaali.

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
        conn.commit()


def insert_listing(rec):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO listings
            (title, description, city, district, price, bedrooms, bathrooms, area, furnished, created_at, image_blob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )
        conn.commit()


def load_listings_df():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        df = pd.read_sql_query("SELECT * FROM listings ORDER BY created_at DESC", conn)
    # Columns types
    if not df.empty:
        df["furnished"] = df["furnished"].astype(int)
        df["bedrooms"] = df["bedrooms"].astype(int)
        df["bathrooms"] = df["bathrooms"].astype(int)
        df["price"] = df["price"].astype(float)
        df["area"] = df["area"].astype(float)
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
    # samee text isku-dhaf ah si similarity u noqoto macno
    corpus = (
        df[["title", "description", "city", "district"]]
        .fillna("")
        .agg(" ".join, axis=1)
        .tolist()
    )
    vectorizer = TfidfVectorizer(min_df=1, stop_words=None)
    mat = vectorizer.fit_transform(corpus + [query_text])
    sims = cosine_similarity(mat[-1], mat[:-1]).flatten()
    df = df.copy()
    df["similarity"] = sims
    return df.sort_values("similarity", ascending=False).head(top_n)


# ---------- AI: Price Estimator ----------

def train_price_model(df: pd.DataFrame):
    if len(df) < 4:
        return None  # xogtu aad bay u yar tahay
    X = df[["city", "district", "bedrooms", "bathrooms", "area", "furnished"]]
    y = df["price"]
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["city", "district"]),
            ("num", "passthrough", ["bedrooms", "bathrooms", "area", "furnished"]),
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
    # Halkan waxa lagu beddeli karaa LLM API (OpenAI iwm) haddii la rabo
    msg = message.lower()
    hints = []
    if any(k in msg for k in ["qiime", "price", "kireyn", "kirada"]):
        hints.append("Waxaad isticmaalikartaa qaybta 'Qiyaas Qiime' si aad u aragto qiime suurtagal ah.")
    if any(k in msg for k in ["raadin", "raadso", "guri", "apartment", "degmo", "city"]):
        hints.append("Tag 'Raadi Guri' oo isticmaal shaandhayn si aad u hesho guri kugu habboon.")
    if any(k in msg for k in ["diiwaan", "ku dar", "listing"]):
        hints.append("Ku dar guri cusub adigoo buuxinaya foomka 'Ku dar Guri'.")
    if not hints:
        hints.append("Waa ku salaaman tahay! Iigu sheeg magaalada, degmadda, iyo miisaaniyadda — waan ku hagayaa.")
    return "\n\n".join(hints)


# ---------- UI ----------

st.set_page_config(page_title="GuriKire AI", page_icon="🏠", layout="wide")
init_db()

st.sidebar.title("🏠 GuriKire AI")
page = st.sidebar.radio("Dooro Bogga:", ["Raadi Guri", "Ku dar Guri", "Qiyaas Qiime", "Caawiye"], index=0)

st.sidebar.markdown("— **MVP**: xogtu waxa lagu kaydiyaa SQLite gudaha app-ka.")

if page == "Ku dar Guri":
    st.header("Ku dar Guri Kire ah")
    with st.form("add_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            title = st.text_input("Cinwaan / Nooc (tusaale: 2-qol Apartment)")
            city = st.text_input("Magaalo")
            district = st.text_input("Degmo / Xaafad")
            price = st.number_input("Qiimaha kirada (USD ama ETB)", min_value=0.0, step=10.0)
            area = st.number_input("Cabbirka (m²)", min_value=0.0, step=1.0)
        with col2:
            bedrooms = st.number_input("Qolal jiif", min_value=0, step=1)
            bathrooms = st.number_input("Musqulo", min_value=0, step=1)
            furnished = st.checkbox("Furnitured / Alaab yaal")
            image_file = st.file_uploader("Ku dar sawir (JPEG/PNG)", type=["jpg", "jpeg", "png"]) 
        description = st.text_area("Sharaxaad faahfaahsan")
        submitted = st.form_submit_button("💾 Kaydi Guri")
        if submitted:
            if not title or not city:
                st.error("Fadlan geli ugu yaraan Cinwaan iyo Magaalo.")
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
                })
                st.success("Waa la kaydiyay! 🟢")

elif page == "Raadi Guri":
    st.header("Raadi Guri la kireysan karo")
    df = load_listings_df()

    with st.expander("🔎 Shaandhayn"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            q_city = st.text_input("Magaalo")
        with c2:
            min_price, max_price = st.slider("Qiime (min, max)", 0, 100000, (0, 100000))
        with c3:
            min_bed = st.number_input("Qolal ugu yaraan", min_value=0, step=1)
        with c4:
            furnished_only = st.checkbox("Kaliya kuwa alaab yaal")
        q_text = st.text_input("Waxaad rabtaa noocee? (qoraal kooban)")

    df_f = df.copy()
    if q_city:
        df_f = df_f[df_f["city"].str.contains(q_city, case=False, na=False)]
    df_f = df_f[(df_f["price"] >= min_price) & (df_f["price"] <= max_price)]
    df_f = df_f[df_f["bedrooms"] >= int(min_bed)]
    if furnished_only:
        df_f = df_f[df_f["furnished"] == 1]

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
                    img = blob_to_image(row["image_blob"]) if not pd.isna(row["image_blob"]) else None
                    if img is not None:
                        st.image(img, caption=row["title"], use_column_width=True)
                    else:
                        st.markdown("*(Ma jiro sawir la dhigay)*")
                with c2:
                    st.subheader(row["title"]) 
                    st.markdown(f"**Goob:** {row['city'] or ''} — {row['district'] or ''}")
                    st.markdown(f"**Qiime:** {row['price']}")
                    st.markdown(f"**Qolal/Musqulo:** {int(row['bedrooms'])} / {int(row['bathrooms'])}")
                    st.markdown(f"**Cabbir:** {row['area']} m² · **Alaab:** {'Haa' if int(row['furnished'])==1 else 'Maya'}")
                    if "similarity" in row:
                        st.progress(float(row["similarity"]))
                    if row["description"]:
                        st.write(row["description"])

elif page == "Qiyaas Qiime":
    st.header("Qiyaas qiimaha kirada")
    df = load_listings_df()
    model = train_price_model(df) if not df.empty else None

    c1, c2, c3 = st.columns(3)
    with c1:
        city = st.text_input("Magaalo", value=df["city"].iloc[0] if not df.empty else "")
        district = st.text_input("Degmo / Xaafad", value=df["district"].iloc[0] if not df.empty else "")
        area = st.number_input("Cabbirka (m²)", min_value=0.0, step=1.0, value=float(df["area"].median()) if not df.empty else 80.0)
    with c2:
        bedrooms = st.number_input("Qolal jiif", min_value=0, step=1, value=int(df["bedrooms"].median()) if not df.empty else 2)
        bathrooms = st.number_input("Musqulo", min_value=0, step=1, value=int(df["bathrooms"].median()) if not df.empty else 1)
    with c3:
        furnished = st.checkbox("Furnitured / Alaab yaal", value=True if not df.empty and df["furnished"].mean() >= 0.5 else False)
        st.markdown("\n")
        run = st.button("🔮 Qiyaas Hadda")

    if run:
        sample = {
            "city": city or "",
            "district": district or "",
            "bedrooms": int(bedrooms),
            "bathrooms": int(bathrooms),
            "area": float(area),
            "furnished": 1 if furnished else 0,
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
