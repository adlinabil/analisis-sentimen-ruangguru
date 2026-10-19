import os
import re
import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, roc_curve, roc_auc_score)
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ------------------------------------------------------------------
# Konfigurasi halaman
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Analisis Sentimen Ruangguru",
    page_icon="📚",
    layout="wide",
)

SEED = 42
N_FOLD = 5
DATA_PATH = "Adli Nabil Mubarok_Test_Analisis Sentimen.csv"
STEM_CACHE_FILE = "stem_cache.json"
PALETTE = {"Negatif": "#EF4444", "Netral": "#F59E0B", "Positif": "#10B981"}
ORDER = ["Negatif", "Netral", "Positif"]
# Hasil GridSearchCV terbaik dari notebook
BEST = {"ngram_range": (1, 1), "max_features": 2000, "min_df": 1, "alpha": 0.5}

sns.set_theme(style="whitegrid")

# ------------------------------------------------------------------
# Preprocessing (sama dengan notebook)
# ------------------------------------------------------------------
STOPWORDS = {
    "di", "dari", "ke", "pada", "dengan", "untuk", "oleh", "tentang", "kepada", "yang", "dan", "atau", "tetapi", "tapi",
    "karena", "meskipun", "jika", "agar", "saya", "aku", "kamu", "dia", "kita", "mereka", "ini", "itu", "lah", "tuh", "nya",
    "dong", "aja", "gitu", "deh", "emang", "ya", "sih", "kok", "lho", "nih", "si", "pun", "juga", "sudah", "akan", "lagi",
    "pernah", "hanya", "adalah", "dalam", "ada", "para", "sebagai", "saat", "ketika", "kalau", "kalo", "jadi", "sama", "mau",
}
NEGATIONS = {"tidak", "bukan", "belum", "jangan", "kurang", "gak", "ga", "nggak", "ngga", "tak", "gk", "enggak"}
DOMAIN_STOPWORDS = {"ruangguru", "rg", "ruang", "kak"}
SLANG = {
    "gak": "tidak", "ga": "tidak", "nggak": "tidak", "ngga": "tidak", "gk": "tidak", "enggak": "tidak", "tdk": "tidak", "tak": "tidak",
    "bgt": "banget", "bgus": "bagus", "bagu": "bagus", "bagust": "bagus", "mantul": "mantap", "udh": "sudah", "udah": "sudah",
    "sdh": "sudah", "blm": "belum", "yg": "yang", "dgn": "dengan", "krn": "karena", "karna": "karena", "utk": "untuk", "jd": "jadi",
    "tp": "tapi", "tpi": "tapi", "klo": "kalau", "kalo": "kalau", "sy": "saya", "sm": "sama", "sma": "sama", "bs": "bisa", "bsa": "bisa",
    "gabisa": "tidak bisa", "lemot": "lambat", "lag": "lambat", "ngelag": "lambat", "eror": "error", "erorr": "error",
    "apk": "aplikasi", "app": "aplikasi", "aplikasinya": "aplikasi", "thx": "terima kasih", "makasih": "terima kasih",
    "terimakasih": "terima kasih", "trims": "terima kasih", "oke": "ok", "okey": "ok", "recomended": "recommended",
    "rekomen": "recommended", "kereen": "keren", "jelek": "buruk", "jlek": "buruk", "pake": "pakai", "pke": "pakai",
    "jg": "juga", "dr": "dari", "dl": "dulu", "bgmn": "bagaimana", "gmn": "bagaimana", "knp": "kenapa", "kyk": "seperti",
    "bnyk": "banyak", "bebrpa": "beberapa",
}

ASPEK = {
    "Mentor / Guru": r"guru|mentor|pengajar|tutor|kakak|penjelasan|menjelaskan",
    "Materi & Soal": r"materi|soal|latihan|rangkuman|pembahasan|kuis|pelajaran|kurikulum|bab",
    "Video & Animasi": r"video|animasi|tayang|nonton|subtitle|suara",
    "Harga & Langganan": r"harga|mahal|langganan|berlangganan|paket|bayar|biaya|promo|diskon|gratis|saldo|refund",
    "Teknis & Performa": r"error|bug|crash|lemot|lambat|loading|download|kuota|login|update|hang|ngelag|tidak bisa|gak bisa|susah",
    "Fitur & Tampilan": r"fitur|tampilan|menu|interface|desain|navigasi|opsi",
    "Layanan & Iklan": r"spam|iklan|telepon|sales|cs|customer|layanan|notifikasi|tawar",
}


@st.cache_resource
def get_stemmer():
    return StemmerFactory().create_stemmer()


def beri_label(score):
    if score <= 2:
        return "Negatif"
    if score == 3:
        return "Netral"
    return "Positif"


def bersihkan_teks(teks, stem_cache, stemming=True):
    stemmer = get_stemmer()
    t = str(teks).lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"\bruang\s+guru\b", "ruangguru", t)
    t = re.sub(r"(.)\1{2,}", r"\1", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    tokens = " ".join(SLANG.get(w, w) for w in t.split()).split()
    tokens = [w for w in tokens if (w not in STOPWORDS or w in NEGATIONS) and w not in DOMAIN_STOPWORDS]
    tokens = [w for w in tokens if len(w) > 1]
    if stemming:
        out = []
        for w in tokens:
            if w not in stem_cache:
                stem_cache[w] = stemmer.stem(w)
            out.append(stem_cache[w])
        tokens = out
    return " ".join(tokens)


def buat_sampler(y, faktor=1.0):
    min_n = int(pd.Series(y).value_counts().min() * faktor)
    k = min(5, min_n - 1)
    return SMOTE(k_neighbors=k, random_state=SEED) if k >= 1 else RandomOverSampler(random_state=SEED)


# ------------------------------------------------------------------
# Load data + latih model (di-cache agar tidak diulang)
# ------------------------------------------------------------------
@st.cache_data(show_spinner="Memuat data & preprocessing (pertama kali agak lama karena stemming)...")
def load_data(path):
    df = pd.read_csv(path, encoding="utf-8")
    df = df.dropna(subset=["content", "score"]).copy()
    df["at"] = pd.to_datetime(df["at"], errors="coerce")
    df["label"] = df["score"].apply(beri_label)

    stem_cache = {}
    if os.path.exists(STEM_CACHE_FILE):
        try:
            stem_cache = json.load(open(STEM_CACHE_FILE, encoding="utf-8"))
        except Exception:
            stem_cache = {}
    df["clean_content"] = df["content"].apply(lambda x: bersihkan_teks(x, stem_cache))
    try:
        json.dump(stem_cache, open(STEM_CACHE_FILE, "w", encoding="utf-8"), ensure_ascii=False)
    except OSError:
        pass
    df = df[df["clean_content"].str.strip().str.len() > 0].reset_index(drop=True)
    df["panjang_kata"] = df["content"].astype(str).str.split().str.len()
    return df


@st.cache_resource(show_spinner="Melatih model Naive Bayes...")
def train_model(_df, n_rows):
    df = _df
    X_tr_txt, X_te_txt, y_tr, y_te = train_test_split(
        df["clean_content"], df["label"], test_size=0.2, stratify=df["label"], random_state=SEED
    )
    vec = TfidfVectorizer(ngram_range=BEST["ngram_range"], max_features=BEST["max_features"],
                          min_df=BEST["min_df"], sublinear_tf=True)
    X_tr = vec.fit_transform(X_tr_txt)
    X_te = vec.transform(X_te_txt)
    X_bal, y_bal = buat_sampler(y_tr).fit_resample(X_tr, y_tr)

    model = MultinomialNB(alpha=BEST["alpha"]).fit(X_bal, y_bal)
    classes = list(model.classes_)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)

    metrik = {
        "Accuracy": accuracy_score(y_te, y_pred),
        "Precision (macro)": precision_score(y_te, y_pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_te, y_pred, average="macro", zero_division=0),
        "F1-Score (macro)": f1_score(y_te, y_pred, average="macro", zero_division=0),
        "F1-Score (weighted)": f1_score(y_te, y_pred, average="weighted", zero_division=0),
    }
    report = pd.DataFrame(classification_report(y_te, y_pred, zero_division=0, output_dict=True)).T

    pipe = ImbPipeline([
        ("tfidf", TfidfVectorizer(ngram_range=BEST["ngram_range"], min_df=BEST["min_df"],
                                  max_features=BEST["max_features"], sublinear_tf=True)),
        ("smote", buat_sampler(y_tr, faktor=(N_FOLD - 1) / N_FOLD)),
        ("nb", MultinomialNB(alpha=BEST["alpha"])),
    ])
    skf = StratifiedKFold(N_FOLD, shuffle=True, random_state=SEED)
    cv_acc = cross_val_score(pipe, X_tr_txt, y_tr, cv=skf, scoring="accuracy")
    cv_f1 = cross_val_score(pipe, X_tr_txt, y_tr, cv=skf, scoring="f1_macro")

    return dict(vec=vec, model=model, classes=classes, X_te_txt=X_te_txt, y_te=y_te, y_pred=y_pred,
                y_prob=y_prob, metrik=metrik, report=report, cv_acc=cv_acc, cv_f1=cv_f1,
                n_train=len(y_tr), n_test=len(y_te))


def prediksi(teks, art, top_k=6):
    vec, model, classes = art["vec"], art["model"], art["classes"]
    fitur = np.array(vec.get_feature_names_out())
    flp = model.feature_log_prob_
    bersih = bersihkan_teks(teks, st.session_state.setdefault("_stem", {}))
    X = vec.transform([bersih])
    p = model.predict_proba(X)[0]
    pred = classes[p.argmax()]
    ref = classes[np.argsort(p)[::-1][1]]
    kontrib = [(fitur[j], (flp[classes.index(pred), j] - flp[classes.index(ref), j]) * X[0, j])
               for j in X.nonzero()[1]]
    kontrib = sorted(kontrib, key=lambda x: -x[1])[:top_k]
    return bersih, pred, dict(zip(classes, p)), kontrib


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("📚 Analisis Sentimen Ulasan Aplikasi Ruangguru")
st.markdown(
    """
    Aplikasi ini menganalisis **ulasan Google Play Store aplikasi Ruangguru** dan memprediksi sentimennya
    (**Negatif / Netral / Positif**) menggunakan **TF-IDF + SMOTE + Multinomial Naive Bayes**.
    Label sentimen dibuat dari rating: 1–2 = Negatif, 3 = Netral, 4–5 = Positif.
    """
)

if not os.path.exists(DATA_PATH):
    st.error(f"File data '{DATA_PATH}' tidak ditemukan. Taruh file CSV di folder yang sama dengan app.py.")
    st.stop()

df = load_data(DATA_PATH)
art = train_model(df, len(df))
classes = art["classes"]

tab_data, tab_eda, tab_eval, tab_fitur, tab_aspek, tab_pred = st.tabs(
    ["📊 Data", "☁️ Eksplorasi Kata", "🎯 Evaluasi Model", "🔑 Fitur Berpengaruh",
     "🧩 Analisis Aspek", "✍️ Prediksi"]
)

# ------------------------------------------------------------------
# Tab 1: Data
# ------------------------------------------------------------------
with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Ulasan", f"{len(df):,}")
    c2.metric("Rata-rata Rating", f"{df['score'].mean():.2f}")
    c3.metric("Rentang Tanggal", f"{df['at'].min():%b %Y} – {df['at'].max():%b %Y}")
    c4.metric("Rata-rata Panjang (kata)", f"{df['panjang_kata'].mean():.1f}")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3.8))
        dist = df["label"].value_counts().reindex(ORDER)
        sns.barplot(x=dist.index, y=dist.values, palette=PALETTE, ax=ax)
        for i, v in enumerate(dist.values):
            ax.text(i, v, int(v), ha="center", va="bottom")
        ax.set_title("Jumlah per Sentimen"); ax.set_xlabel(""); ax.set_ylabel("Jumlah")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 3.8))
        ax.pie(dist, labels=ORDER, autopct="%1.1f%%", colors=[PALETTE[o] for o in ORDER],
               wedgeprops=dict(width=.45))
        ax.set_title("Proporsi Sentimen")
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(5, 3.8))
        sns.countplot(data=df, x="score", palette="viridis", ax=ax)
        ax.set_title("Distribusi Rating")
        st.pyplot(fig)
    with col4:
        fig, ax = plt.subplots(figsize=(5, 3.8))
        yr = df.dropna(subset=["at"]).groupby(df["at"].dt.year).agg(jumlah=("score", "size"), rating=("score", "mean"))
        ax.bar(yr.index, yr["jumlah"], color="#A78BFA")
        ax2 = ax.twinx(); ax2.plot(yr.index, yr["rating"], color="#EF4444", marker="o")
        ax2.set_ylim(1, 5); ax2.set_ylabel("Rata-rata rating")
        ax.set_title("Jumlah Ulasan & Rating per Tahun")
        st.pyplot(fig)

    with st.expander("Lihat data ulasan"):
        f = st.multiselect("Filter sentimen", ORDER, default=ORDER)
        st.dataframe(df[df["label"].isin(f)][["userName", "score", "at", "content", "clean_content", "label"]],
                     use_container_width=True)

# ------------------------------------------------------------------
# Tab 2: Eksplorasi kata
# ------------------------------------------------------------------
with tab_eda:
    st.subheader("WordCloud per Sentimen")
    cmaps = {"Negatif": "Reds", "Netral": "Oranges", "Positif": "Greens"}
    cols = st.columns(3)
    for col, lab in zip(cols, ORDER):
        teks = " ".join(df.loc[df["label"] == lab, "clean_content"])
        with col:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            wc = WordCloud(width=700, height=450, background_color="white", colormap=cmaps[lab],
                           max_words=80, random_state=SEED).generate(teks)
            ax.imshow(wc, interpolation="bilinear"); ax.axis("off"); ax.set_title(lab)
            st.pyplot(fig)

    st.subheader("12 Kata Teratas per Sentimen")
    cols = st.columns(3)
    for col, lab in zip(cols, ORDER):
        top = pd.Series(" ".join(df.loc[df["label"] == lab, "clean_content"]).split()).value_counts().head(12)[::-1]
        with col:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.barh(top.index, top.values, color=PALETTE[lab]); ax.set_title(lab)
            st.pyplot(fig)

# ------------------------------------------------------------------
# Tab 3: Evaluasi
# ------------------------------------------------------------------
with tab_eval:
    st.caption(f"Data latih: {art['n_train']} • Data uji: {art['n_test']} • "
               f"Parameter: alpha={BEST['alpha']}, max_features={BEST['max_features']}, "
               f"min_df={BEST['min_df']}, ngram={BEST['ngram_range']}")
    m = art["metrik"]
    cols = st.columns(len(m))
    for c, (k, v) in zip(cols, m.items()):
        c.metric(k, f"{v:.3f}")

    st.markdown("**Classification Report**")
    st.dataframe(art["report"].style.format("{:.3f}"), use_container_width=True)

    y_te, y_pred, y_prob = art["y_te"], art["y_pred"], art["y_prob"]
    cm = confusion_matrix(y_te, y_pred, labels=classes)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set(title="Confusion Matrix (jumlah)", xlabel="Prediksi", ylabel="Aktual")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set(title="Confusion Matrix (% per kelas aktual)", xlabel="Prediksi", ylabel="Aktual")
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        Y = label_binarize(y_te, classes=classes)
        fig, ax = plt.subplots(figsize=(5, 4))
        aucs = {}
        for i, c in enumerate(classes):
            fpr, tpr, _ = roc_curve(Y[:, i], y_prob[:, i])
            aucs[c] = roc_auc_score(Y[:, i], y_prob[:, i])
            ax.plot(fpr, tpr, lw=2.5, color=PALETTE[c], label=f"{c} (AUC = {aucs[c]:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve (One-vs-Rest)")
        ax.legend(loc="lower right")
        st.pyplot(fig)
        st.caption(f"ROC-AUC macro: {np.mean(list(aucs.values())):.4f}")
    with col4:
        cv_df = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(N_FOLD)],
                              "Accuracy": art["cv_acc"], "F1-macro": art["cv_f1"]}).set_index("Fold")
        fig, ax = plt.subplots(figsize=(5, 4))
        cv_df.plot(kind="bar", ax=ax, color=["#7C3AED", "#EC4899"], rot=0)
        ax.set_ylim(0, 1); ax.set_title("Cross-Validation (5-fold)")
        st.pyplot(fig)
        st.caption(f"Accuracy {art['cv_acc'].mean():.4f} ± {art['cv_acc'].std():.4f} • "
                   f"F1-macro {art['cv_f1'].mean():.4f} ± {art['cv_f1'].std():.4f}")

    st.markdown("**Analisis error** (prediksi salah dengan keyakinan tertinggi)")
    salah = pd.DataFrame({"ulasan (bersih)": art["X_te_txt"].values, "aktual": y_te.values,
                          "prediksi": y_pred, "keyakinan": y_prob.max(axis=1)})
    salah = salah[salah["aktual"] != salah["prediksi"]].sort_values("keyakinan", ascending=False)
    st.write(f"Jumlah salah prediksi: **{len(salah)}** dari {len(y_te)}")
    st.dataframe(salah.head(10), use_container_width=True)

# ------------------------------------------------------------------
# Tab 4: Fitur berpengaruh
# ------------------------------------------------------------------
with tab_fitur:
    top_n = st.slider("Jumlah fitur teratas", 5, 25, 15)
    fitur = np.array(art["vec"].get_feature_names_out())
    flp = art["model"].feature_log_prob_
    cols = st.columns(3)
    for i, (col, c) in enumerate(zip(cols, classes)):
        selisih = flp[i] - np.delete(flp, i, axis=0).mean(axis=0)
        idx = np.argsort(selisih)[::-1][:top_n]
        with col:
            fig, ax = plt.subplots(figsize=(5, 0.3 * top_n + 1.5))
            ax.barh(fitur[idx][::-1], selisih[idx][::-1], color=PALETTE[c])
            ax.set_title(f"Fitur Berpengaruh – {c}"); ax.set_xlabel("Skor pengaruh")
            st.pyplot(fig)

# ------------------------------------------------------------------
# Tab 5: Aspek
# ------------------------------------------------------------------
with tab_aspek:
    low = df["content"].str.lower()
    rows = []
    for nama, pola in ASPEK.items():
        mk = low.str.contains(pola, regex=True); sub = df[mk]
        rows.append({"Aspek": nama, "Jumlah": int(mk.sum()), "Rata-rata Rating": sub["score"].mean(),
                     "% Negatif": (sub["label"] == "Negatif").mean() * 100,
                     "% Netral": (sub["label"] == "Netral").mean() * 100,
                     "% Positif": (sub["label"] == "Positif").mean() * 100})
    asp = pd.DataFrame(rows).sort_values("Jumlah", ascending=False).reset_index(drop=True)
    st.dataframe(asp.style.format({"Rata-rata Rating": "{:.2f}", "% Negatif": "{:.1f}",
                                   "% Netral": "{:.1f}", "% Positif": "{:.1f}"}), use_container_width=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    asp.set_index("Aspek")[["% Negatif", "% Netral", "% Positif"]].plot(
        kind="barh", stacked=True, ax=ax, color=[PALETTE[o] for o in ORDER])
    ax.set_xlabel("% ulasan"); ax.set_title("Komposisi Sentimen per Aspek"); ax.legend(loc="lower right")
    st.pyplot(fig)

# ------------------------------------------------------------------
# Tab 6: Prediksi
# ------------------------------------------------------------------
with tab_pred:
    st.subheader("Prediksi Sentimen Ulasan Baru")
    contoh = {
        "(ketik sendiri)": "",
        "Positif": "Aplikasinya bagus banget, materi lengkap dan gurunya jelas!",
        "Negatif": "Sering error, tidak bisa login, kecewa sekali",
        "Netral": "Lumayan lah tapi masih banyak yang perlu diperbaiki",
    }
    pilih = st.selectbox("Contoh ulasan", list(contoh.keys()))
    teks = st.text_area("Tulis ulasan", value=contoh[pilih], height=120,
                        placeholder="Contoh: Aplikasinya membantu banget untuk belajar...")

    if st.button("🔍 Prediksi Sentimen", type="primary", use_container_width=True):
        if not teks.strip():
            st.warning("Silakan isi teks ulasan terlebih dahulu.")
        else:
            bersih, pred, prob, kontrib = prediksi(teks, art)
            if not bersih.strip():
                st.warning("Teks tidak mengandung kata yang dapat dianalisis setelah preprocessing.")
            else:
                msg = f"Sentimen: **{pred.upper()}** — keyakinan {prob[pred]*100:.1f}%"
                {"Positif": st.success, "Negatif": st.error, "Netral": st.warning}[pred](msg)

                col1, col2 = st.columns(2)
                with col1:
                    pdf = pd.DataFrame({"Sentimen": classes, "Probabilitas (%)": [prob[c] * 100 for c in classes]})
                    st.bar_chart(pdf.set_index("Sentimen"))
                with col2:
                    st.markdown("**Kata pendorong prediksi:**")
                    st.write(", ".join(f"`{k}`" for k, _ in kontrib) or "-")
                    st.markdown("**Teks setelah preprocessing:**")
                    st.code(bersih)

st.divider()
st.caption("Model: TF-IDF + SMOTE + Multinomial Naive Bayes • Data: ulasan Ruangguru (Google Play) • Dibuat dengan Streamlit")