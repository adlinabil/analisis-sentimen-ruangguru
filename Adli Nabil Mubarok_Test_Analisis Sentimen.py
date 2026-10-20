from __future__ import annotations
import io
import json
import os
import re
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                            precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from wordcloud import WordCloud

# Folder = tempat file ini berada. Untuk folder lain, isi manual, mis.:
# BASE_DIR = "/Users/murniramadhani/Documents/Kuliah Analisis Big Data /Data Sentimen"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE, DATA_FILE, STEM_FILE = "model_nb_sentimen.joblib", "data_bersih_berlabel.csv", "stem_cache.json"
SEED, TEST_SIZE = 42, 0.2      # HARUS sama dengan notebook agar data uji identik
COLORS = {"Positif": "#10B981", "Netral": "#F59E0B", "Negatif": "#EF4444"}

st.set_page_config(page_title="Sentiment Mining · Ruangguru", page_icon="💬", layout="wide")
st.markdown("""
<style>
.hero{background:linear-gradient(120deg,#4F46E5 0%,#7C3AED 55%,#EC4899 100%);padding:26px 32px;
    border-radius:18px;color:white;margin-bottom:16px}
.hero h1{margin:0;font-size:1.9rem;color:white} .hero p{margin:6px 0 0;opacity:.92}
.kpi{background:var(--secondary-background-color);border-radius:14px;padding:14px 18px;border-left:6px solid #7C3AED}
.kpi .v{font-size:1.65rem;font-weight:700} .kpi .l{font-size:.78rem;opacity:.7;text-transform:uppercase;letter-spacing:.05em}
.insight{background:var(--secondary-background-color);border-radius:12px;padding:10px 16px;margin:6px 0;border-left:4px solid #EC4899}
.stTabs [data-baseweb="tab"]{font-weight:600}
</style>
<div class="hero"><h1>💬 Sentiment Mining Dashboard · Ruangguru</h1>
<p>Hasil analisis notebook dengan <b>Multinomial Naive Bayes</b> · TF-IDF · SMOTE · Influential Features · Analisis Aspek</p></div>
""", unsafe_allow_html=True)
def kpi(col, label, value, color="#7C3AED"):
    col.markdown(f'<div class="kpi" style="border-left-color:{color}"><div class="l">{label}</div>'
                f'<div class="v">{value}</div></div>', unsafe_allow_html=True)
def _find(name):
    p = os.path.join(BASE_DIR, name)
    return p if os.path.exists(p) else None
with st.sidebar:
    st.header(" Hasil Notebook")
    up_model = st.file_uploader("Model (.joblib) – opsional", type="joblib",
                                help="Isi hanya jika file tidak ada di folder aplikasi.")
    up_data = st.file_uploader("Data bersih (.csv) – opsional", type="csv")
model_src = up_model.getvalue() if up_model else _find(MODEL_FILE)
data_src = up_data.getvalue() if up_data else _find(DATA_FILE)
if model_src is None or data_src is None:
    st.error("File hasil notebook belum ditemukan.")
    st.markdown(f"""
Letakkan file berikut **satu folder dengan `app_streamlit.py`** (folder saat ini: `{BASE_DIR}`):
- `{MODEL_FILE}`
- `{DATA_FILE}`
File ini dibuat otomatis oleh **sel terakhir (bagian 11 – Simpan Model)** di notebook. Jalankan notebook sampai sel itu,
atau unggah kedua file lewat sidebar.
""")
    st.stop()
@st.cache_resource(show_spinner="Memuat model...")
def load_bundle(src):
    return joblib.load(io.BytesIO(src) if isinstance(src, bytes) else src)
@st.cache_data(show_spinner="Memuat data...")
def load_data(src):
    d = pd.read_csv(io.BytesIO(src) if isinstance(src, bytes) else src)
    d["clean_content"] = d["clean_content"].fillna("").astype(str)
    d["at"] = pd.to_datetime(d.get("at"), errors="coerce") if "at" in d.columns else pd.NaT
    return d
bundle = load_bundle(model_src)
df = load_data(data_src)
vec, model = bundle["vectorizer"], bundle["model"]
classes = list(model.classes_)
bp = bundle.get("best_params", {})
cmap = {c: COLORS.get(c, "#6366F1") for c in classes}
fitur = np.array(vec.get_feature_names_out())
flp = model.feature_log_prob_
@st.cache_resource(show_spinner="Menghitung evaluasi pada data uji...")
def build_eval(_key, _df, _vec, _model):
    """Membentuk ulang data uji yang SAMA dengan notebook (split deterministik) lalu hanya memprediksi."""
    X_tr, X_te, y_tr, y_te = train_test_split(_df["clean_content"], _df["label"], test_size=TEST_SIZE,
                                            stratify=_df["label"], random_state=SEED)
    cls = list(_model.classes_)
    y_pred = _model.predict(_vec.transform(X_te))
    prob = _model.predict_proba(_vec.transform(X_te))

    # ROC One-vs-Rest
    roc = {}
    if len(cls) == 2:
        yb = (y_te == cls[1]).astype(int)
        f, t, _ = roc_curve(yb, prob[:, 1]); roc[cls[1]] = (f, t, roc_auc_score(yb, prob[:, 1]))
    else:
        Y = label_binarize(y_te, classes=cls)
        for i, c in enumerate(cls):
            f, t, _ = roc_curve(Y[:, i], prob[:, i]); roc[c] = (f, t, roc_auc_score(Y[:, i], prob[:, i]))

    metrics = {
        "Accuracy": accuracy_score(y_te, y_pred),
        "Precision (macro)": precision_score(y_te, y_pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_te, y_pred, average="macro", zero_division=0),
        "F1-Score (macro)": f1_score(y_te, y_pred, average="macro", zero_division=0),
        "F1-Score (weighted)": f1_score(y_te, y_pred, average="weighted", zero_division=0),
        "ROC-AUC (macro)": float(np.mean([v[2] for v in roc.values()])),
    }
    rep = pd.DataFrame(classification_report(y_te, y_pred, output_dict=True, zero_division=0)).T
    errors = pd.DataFrame({"ulasan": X_te.values, "aktual": y_te.values, "prediksi": y_pred, "keyakinan": prob.max(axis=1)})
    errors = errors[errors["aktual"] != errors["prediksi"]].sort_values("keyakinan", ascending=False)
    return dict(classes=cls, metrics=metrics, report=rep, cm=confusion_matrix(y_te, y_pred, labels=cls), roc=roc,
                errors=errors, X_tr=X_tr, y_tr=y_tr, n_tr=len(X_tr), n_te=len(X_te), y_te=y_te, y_pred=y_pred)
ev = build_eval(f"{len(df)}-{len(fitur)}-{bundle.get('metrics', {}).get('Accuracy', 0):.6f}", df, vec, model)
M = ev["metrics"]
saved_acc = bundle.get("metrics", {}).get("Accuracy")
in_sync = saved_acc is None or abs(saved_acc - M["Accuracy"]) < 1e-9
with st.sidebar:
    st.divider()
    st.subheader(" Model")
    st.markdown(f"""
- Algoritma: **Multinomial Naive Bayes**
- alpha: `{bp.get('nb__alpha', model.alpha)}`
- n-gram: `{bp.get('tfidf__ngram_range', vec.ngram_range)}`
- max_features: `{bp.get('tfidf__max_features', vec.max_features)}`
- min_df: `{bp.get('tfidf__min_df', vec.min_df)}`
- Fitur TF-IDF: `{len(fitur):,}`
- Data latih / uji: `{ev['n_tr']}` / `{ev['n_te']}`
""")
    if in_sync:
        st.success("Hasil identik dengan notebook ")
    else:
        st.warning("Akurasi berbeda dari yang tersimpan di notebook. Pastikan model & data berasal dari proses yang sama.")
# =============================================================================
# 2. FUNGSI BANTU (preprocessing HANYA untuk teks baru pada tab Prediksi)
# =============================================================================
STOPWORDS = {
    "di", "dari", "ke", "pada", "dengan", "untuk", "oleh", "tentang", "kepada", "yang", "dan", "atau", "tetapi", "tapi",
    "karena", "meskipun", "jika", "agar", "saya", "aku", "kamu", "dia", "kita", "mereka", "ini", "itu", "lah", "tuh", "nya",
    "dong", "aja", "gitu", "deh", "emang", "ya", "sih", "kok", "lho", "nih", "si", "pun", "juga", "sudah", "akan", "lagi",
    "pernah", "hanya", "adalah", "dalam", "ada", "para", "sebagai", "saat", "ketika", "kalau", "kalo", "jadi", "sama", "mau",
}
NEGATIONS = {"tidak", "bukan", "belum", "jangan", "kurang", "gak", "ga", "nggak", "ngga", "tak", "gk", "enggak"}
DOMAIN_STOPWORDS = {"ruangguru", "rg", "ruang", "kak"}
SLANG = {
    "gak": "tidak", "ga": "tidak", "nggak": "tidak", "ngga": "tidak", "gk": "tidak", "enggak": "tidak", "tdk": "tidak",
    "tak": "tidak", "bgt": "banget", "bgus": "bagus", "bagu": "bagus", "bagust": "bagus", "mantul": "mantap", "udh": "sudah",
    "udah": "sudah", "sdh": "sudah", "blm": "belum", "yg": "yang", "dgn": "dengan", "krn": "karena", "karna": "karena",
    "utk": "untuk", "jd": "jadi", "tp": "tapi", "tpi": "tapi", "klo": "kalau", "kalo": "kalau", "sy": "saya", "sm": "sama",
    "sma": "sama", "bs": "bisa", "bsa": "bisa", "gabisa": "tidak bisa", "lemot": "lambat", "lag": "lambat",
    "ngelag": "lambat", "eror": "error", "erorr": "error", "apk": "aplikasi", "app": "aplikasi", "aplikasinya": "aplikasi",
    "thx": "terima kasih", "makasih": "terima kasih", "terimakasih": "terima kasih", "trims": "terima kasih", "oke": "ok",
    "okey": "ok", "recomended": "recommended", "rekomen": "recommended", "kereen": "keren", "jelek": "buruk",
    "jlek": "buruk", "pake": "pakai", "pke": "pakai", "jg": "juga", "dr": "dari", "dl": "dulu", "bgmn": "bagaimana",
    "gmn": "bagaimana", "knp": "kenapa", "kyk": "seperti", "bnyk": "banyak", "bebrpa": "beberapa",
}
_STEM_PATH = os.path.join(BASE_DIR, STEM_FILE)
_stem_cache: dict = {}
if os.path.exists(_STEM_PATH):
    try:
        _stem_cache = json.load(open(_STEM_PATH, encoding="utf-8"))
    except (OSError, ValueError):
        _stem_cache = {}
@st.cache_resource(show_spinner="Memuat stemmer Sastrawi...")
def get_stemmer():
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    except ImportError:
        return None
    return StemmerFactory().create_stemmer()
def stem_kata(w):
    if w not in _stem_cache:
        s = get_stemmer()
        _stem_cache[w] = s.stem(w) if s else w
    return _stem_cache[w]
def bersihkan_teks(teks):
    t = str(teks).lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"\bruang\s+guru\b", "ruangguru", t)
    t = re.sub(r"(.)\1{2,}", r"\1", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    tokens = " ".join(SLANG.get(w, w) for w in t.split()).split()
    tokens = [w for w in tokens if (w not in STOPWORDS or w in NEGATIONS) and w not in DOMAIN_STOPWORDS and len(w) > 1]
    return " ".join(stem_kata(w) for w in tokens)
def simpan_stem_cache():
    try:
        json.dump(_stem_cache, open(_STEM_PATH, "w", encoding="utf-8"), ensure_ascii=False)
    except OSError:
        pass
ASPEK = {
    "Mentor / Guru": r"guru|mentor|pengajar|tutor|kakak|penjelasan|menjelaskan",
    "Materi & Soal": r"materi|soal|latihan|rangkuman|pembahasan|kuis|pelajaran|kurikulum|bab",
    "Video & Animasi": r"video|animasi|tayang|nonton|subtitle|suara",
    "Harga & Langganan": r"harga|mahal|langganan|berlangganan|paket|bayar|biaya|promo|diskon|gratis|saldo|refund",
    "Teknis & Performa": r"error|bug|crash|lemot|lambat|loading|download|kuota|login|update|hang|ngelag|tidak bisa|gak bisa|susah",
    "Fitur & Tampilan": r"fitur|tampilan|menu|interface|desain|navigasi|opsi",
    "Layanan & Iklan": r"spam|iklan|telepon|sales|cs|customer|layanan|notifikasi|tawar",
}
@st.cache_data(show_spinner=False)
def aspect_table(_df, key):
    low = _df["content"].astype(str).str.lower()
    rows = []
    for nama, pola in ASPEK.items():
        m = low.str.contains(pola, regex=True); sub = _df[m]
        rows.append({"Aspek": nama, "Jumlah Ulasan": int(m.sum()), "Porsi (%)": m.mean() * 100,
                     "Rata-rata Rating": sub["score"].mean(),
                      "% Negatif": (sub["label"] == "Negatif").mean() * 100,
                      "% Netral": (sub["label"] == "Netral").mean() * 100,
                     "% Positif": (sub["label"] == "Positif").mean() * 100})
    return pd.DataFrame(rows).sort_values("Jumlah Ulasan", ascending=False).reset_index(drop=True)



def influential(top_n=15):
    rows = []
    for i, c in enumerate(classes):
        diff = flp[i] - np.delete(flp, i, axis=0).mean(axis=0)
        for rank, j in enumerate(np.argsort(diff)[::-1][:top_n], 1):
            rows.append({"Kelas": c, "Rank": rank, "Fitur": fitur[j], "Skor Pengaruh": diff[j],
                        "Rasio Odds": float(np.exp(diff[j])), "log P(kata|kelas)": flp[i, j]})
    return pd.DataFrame(rows)



def prediksi(teks, top_k=8):
    bersih = bersihkan_teks(teks)
    if not bersih.strip():
        return None
    X = vec.transform([bersih])
    p = model.predict_proba(X)[0]
    order = np.argsort(p)[::-1]
    pred, ref = classes[order[0]], classes[order[1]]
    kontrib = [{"Kata": fitur[j], "TF-IDF": float(X[0, j]),
                f"Dorongan ke {pred}": float((flp[classes.index(pred), j] - flp[classes.index(ref), j]) * X[0, j])}
            for j in X.nonzero()[1]]
    cdf = pd.DataFrame(kontrib)
    if not cdf.empty:
        cdf = cdf.sort_values(cdf.columns[-1], ascending=False).head(top_k).reset_index(drop=True)
    return {"clean": bersih, "pred": pred, "proba": dict(zip(classes, p)), "contrib": cdf}



# =============================================================================
# 3. TAMPILAN
# =============================================================================
tabs = st.tabs(["🏠 Ringkasan", "📊 Eksplorasi Data", "☁ Analisis Teks", "🤖 Evaluasi Model",
                 "⭐ Influential Features", "🧩 Aspek & Anomali", "🔮 Prediksi", "📘 Metodologi"])

# ---------------------------------------------------------------- RINGKASAN
with tabs[0]:
    c = st.columns(5)
    for col, (k, v), colr in zip(c, [(k, M[k]) for k in ["Accuracy", "Precision (macro)", "Recall (macro)",
                                                        "F1-Score (macro)", "ROC-AUC (macro)"]],
                                 ["#4F46E5", "#10B981", "#F59E0B", "#EC4899", "#06B6D4"]):
        kpi(col, k, f"{v*100:.1f}%" if "AUC" not in k else f"{v:.3f}", colr)
    st.write("")
    c = st.columns(4)
    kpi(c[0], "Total Ulasan", f"{len(df):,}")
    kpi(c[1], "Rata-rata Rating", f"{df['score'].mean():.2f} ⭐", "#F59E0B")
    kpi(c[2], "% Positif", f"{(df['label']=='Positif').mean()*100:.1f}%", "#10B981")
    kpi(c[3], "% Negatif", f"{(df['label']=='Negatif').mean()*100:.1f}%", "#EF4444")

    st.subheader("💡 Temuan Utama")
    rep = ev["report"].loc[classes]
    best_c, worst_c = rep["f1-score"].idxmax(), rep["f1-score"].idxmin()
    asp = aspect_table(df, len(df))
    neg_asp = asp.sort_values("% Negatif", ascending=False).iloc[0]
    top_asp = asp.iloc[0]
    insights = [
        f"Model paling baik mengenali kelas **{best_c}** (F1 = {rep.loc[best_c, 'f1-score']:.2f}) dan paling lemah pada kelas "
        f"**{worst_c}** (F1 = {rep.loc[worst_c, 'f1-score']:.2f}).",
        f"Aspek yang paling sering dibahas adalah **{top_asp['Aspek']}** ({top_asp['Porsi (%)']:.0f}% ulasan), "
        f"sedangkan aspek paling negatif adalah **{neg_asp['Aspek']}** ({neg_asp['% Negatif']:.0f}% ulasan negatif).",
    ]
    if df["at"].notna().any():
        yr = df.dropna(subset=["at"]).groupby(df["at"].dt.year)["score"].agg(["mean", "size"])
        yr = yr[yr["size"] >= 20]
        if len(yr):
            insights.append(f"Rata-rata rating tertinggi tercatat pada tahun **{int(yr['mean'].idxmax())}** "
                            f"({yr['mean'].max():.2f}) dan terendah pada **{int(yr['mean'].idxmin())}** ({yr['mean'].min():.2f}).")
    imb = df["label"].value_counts(normalize=True)
    insights.append(f"Data **tidak seimbang**: kelas {imb.idxmax()} mendominasi ({imb.max()*100:.0f}%), sehingga "
                    f"**F1-macro ({M['F1-Score (macro)']:.2f})** lebih jujur daripada akurasi ({M['Accuracy']:.2f}).")
    for t in insights:
        st.markdown(f'<div class="insight">{t}</div>', unsafe_allow_html=True)

    a, b = st.columns(2)
    lb = df["label"].value_counts().reset_index(); lb.columns = ["Sentimen", "Jumlah"]
    a.plotly_chart(px.pie(lb, names="Sentimen", values="Jumlah", hole=.55, color="Sentimen",
                          color_discrete_map=cmap, title="Distribusi Sentimen"), width="stretch")
    rp = rep[["precision", "recall", "f1-score"]].reset_index().melt("index", var_name="Metrik", value_name="Nilai")
    b.plotly_chart(px.bar(rp, x="index", y="Nilai", color="Metrik", barmode="group", range_y=[0, 1],
                          title="Kinerja per Kelas (data uji)", labels={"index": "Kelas"}), width="stretch")

# ---------------------------------------------------------------- EKSPLORASI
with tabs[1]:
    d = df.copy()
    f1c, f2c = st.columns(2)
    if d["at"].notna().any():
        yrs = sorted(d["at"].dropna().dt.year.unique())
        yr_sel = f1c.select_slider("Rentang tahun", options=yrs, value=(yrs[0], yrs[-1]))
        d = d[d["at"].dt.year.between(*yr_sel) | d["at"].isna()]
    lab_sel = f2c.multiselect("Sentimen", classes, default=classes)
    d = d[d["label"].isin(lab_sel)]
    st.caption(f"{len(d):,} ulasan terpilih")

    a, b = st.columns(2)
    sc = d["score"].value_counts().sort_index().reset_index(); sc.columns = ["Rating", "Jumlah"]
    a.plotly_chart(px.bar(sc, x="Rating", y="Jumlah", text="Jumlah", color="Rating", color_continuous_scale="Viridis",
                          title="Distribusi Rating"), width="stretch")
    d["panjang_kata"] = d["content"].astype(str).str.split().str.len()
    b.plotly_chart(px.histogram(d, x="panjang_kata", color="label", nbins=30, barmode="overlay", opacity=.7,
                                color_discrete_map=cmap, title="Panjang Ulasan (kata)"), width="stretch")
    if d["at"].notna().any():
        dd = d.dropna(subset=["at"]).copy()
        dd["Tahun"] = dd["at"].dt.year
        a, b = st.columns(2)
        t = dd.groupby(["Tahun", "label"]).size().reset_index(name="Jumlah")
        a.plotly_chart(px.bar(t, x="Tahun", y="Jumlah", color="label", color_discrete_map=cmap,
                              title="Jumlah Ulasan per Tahun"), width="stretch")
        r = dd.groupby("Tahun")["score"].agg(["mean", "size"]).reset_index(); r.columns = ["Tahun", "Rating", "Jumlah"]
        fig = px.line(r, x="Tahun", y="Rating", markers=True, hover_data=["Jumlah"], title="Rata-rata Rating per Tahun")
        fig.update_traces(line_color="#7C3AED"); fig.update_yaxes(range=[1, 5])
        b.plotly_chart(fig, width="stretch")
    st.subheader("Tabel Ulasan")
    q = st.text_input("Cari kata di ulasan")
    tb = d[d["content"].astype(str).str.contains(q, case=False, regex=False)] if q else d
    st.dataframe(tb[[c for c in ["at", "score", "label", "content"] if c in tb.columns]].head(300), width="stretch")

# ---------------------------------------------------------------- TEKS
with tabs[2]:
    pick = st.selectbox("WordCloud untuk", ["Semua"] + classes)
    txt = " ".join(df["clean_content"] if pick == "Semua" else df.loc[df["label"] == pick, "clean_content"])
    if txt.strip():
        cm_name = {"Negatif": "Reds", "Netral": "Oranges", "Positif": "Greens"}.get(pick, "viridis")
        wc = WordCloud(width=1000, height=420, background_color="white", colormap=cm_name, max_words=100,
                       random_state=SEED).generate(txt)
        fig, ax = plt.subplots(figsize=(11, 4.6)); ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        st.pyplot(fig)
    cols = st.columns(len(classes))
    for col, c in zip(cols, classes):
        top = pd.Series(" ".join(df.loc[df["label"] == c, "clean_content"]).split()).value_counts().head(12)[::-1]
        col.plotly_chart(px.bar(x=top.values, y=top.index, orientation="h", title=f"Kata teratas – {c}",
                                color_discrete_sequence=[cmap[c]], labels={"x": "Frekuensi", "y": ""}), width="stretch")

    st.divider()
    st.subheader("🔍 Penelusur Kata")
    kw = st.text_input("Ketik kata/frasa (mis. error, mahal, guru, video)")
    if kw:
        m = df["content"].astype(str).str.contains(kw, case=False, regex=False)
        st.write(f"**{int(m.sum())}** ulasan menyebut “{kw}” · rata-rata rating **{df.loc[m, 'score'].mean():.2f}**")
        if m.sum():
            dist = df.loc[m, "label"].value_counts().reset_index(); dist.columns = ["Sentimen", "Jumlah"]
            a, b = st.columns([1, 2])
            a.plotly_chart(px.pie(dist, names="Sentimen", values="Jumlah", hole=.5, color="Sentimen",
                                  color_discrete_map=cmap), width="stretch")
            b.dataframe(df.loc[m, ["score", "label", "content"]].head(50), width="stretch")

# ---------------------------------------------------------------- EVALUASI
with tabs[3]:
    c = st.columns(6)
    for col, (k, v), colr in zip(c, M.items(), ["#4F46E5", "#10B981", "#F59E0B", "#EC4899", "#8B5CF6", "#06B6D4"]):
        kpi(col, k, f"{v*100:.2f}%" if "AUC" not in k else f"{v:.3f}", colr)
    st.write("")
    a, b = st.columns(2)
    cm = ev["cm"]; cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    mode = a.radio("Tampilan confusion matrix", ["Jumlah", "% per kelas aktual (recall)"], horizontal=True)
    z, fmt = (cm, "d") if mode == "Jumlah" else (cm_pct, ".1f")
    a.plotly_chart(px.imshow(z, x=classes, y=classes, text_auto=fmt, color_continuous_scale="Purples",
                             labels=dict(x="Prediksi", y="Aktual", color=""), title="Confusion Matrix"), width="stretch")
    fig = go.Figure()
    for cl, (fpr, tpr, auc) in ev["roc"].items():
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{cl} (AUC={auc:.3f})", line=dict(width=3, color=cmap.get(cl))))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), showlegend=False))
    fig.update_layout(title="ROC Curve (One-vs-Rest)", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    b.plotly_chart(fig, width="stretch")

    st.subheader("Classification Report")
    st.dataframe(ev["report"].style.format("{:.3f}").background_gradient(cmap="Purples", subset=["precision", "recall", "f1-score"]),
                 width="stretch")

    st.subheader("Cross-Validation (opsional)")
    st.caption("Menghitung ulang TF-IDF + SMOTE + Naive Bayes di tiap fold pada data latih dengan parameter terbaik dari notebook.")
    nf = st.slider("Jumlah fold", 3, 10, 5)
    if st.button("Jalankan Cross-Validation"):
        from imblearn.over_sampling import SMOTE, RandomOverSampler
        from imblearn.pipeline import Pipeline as ImbPipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import StratifiedKFold, cross_validate
        from sklearn.naive_bayes import MultinomialNB
        mn = int(ev["y_tr"].value_counts().min() * (nf - 1) / nf)
        smp = SMOTE(k_neighbors=min(5, mn - 1), random_state=SEED) if mn - 1 >= 1 else RandomOverSampler(random_state=SEED)
        pipe = ImbPipeline([("tfidf", TfidfVectorizer(ngram_range=vec.ngram_range, max_features=vec.max_features,
                                                       min_df=vec.min_df, sublinear_tf=True)),
                              ("smote", smp), ("nb", MultinomialNB(alpha=model.alpha))])
        with st.spinner("Menghitung..."):
            cv = cross_validate(pipe, ev["X_tr"], ev["y_tr"], cv=StratifiedKFold(nf, shuffle=True, random_state=SEED),
                                scoring=["accuracy", "f1_macro"])
        st.session_state["cv"] = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(nf)],
                                               "Accuracy": cv["test_accuracy"], "F1-macro": cv["test_f1_macro"]})
    if "cv" in st.session_state:
        cvd = st.session_state["cv"]
        st.plotly_chart(px.bar(cvd.melt("Fold", var_name="Metrik", value_name="Nilai"), x="Fold", y="Nilai", color="Metrik",
                               barmode="group", range_y=[0, 1],
                               title=f"CV: Accuracy {cvd['Accuracy'].mean():.3f} ± {cvd['Accuracy'].std():.3f} · "
                                     f"F1-macro {cvd['F1-macro'].mean():.3f} ± {cvd['F1-macro'].std():.3f}"), width="stretch")

    with st.expander("❌ Analisis kesalahan prediksi"):
        er = ev["errors"]
        st.write(f"{len(er)} salah prediksi dari {ev['n_te']} data uji")
        pair = er.groupby(["aktual", "prediksi"]).size().reset_index(name="Jumlah").sort_values("Jumlah", ascending=False)
        st.dataframe(pair, width="stretch")
        st.dataframe(er.head(50).style.format({"keyakinan": "{:.2f}"}), width="stretch")

# ---------------------------------------------------------------- INFLUENTIAL FEATURES
with tabs[4]:
    st.subheader("⭐ Influential Features")
    st.caption("Skor pengaruh = log P(kata | kelas) − rata-rata log P(kata | kelas lain). Makin besar, makin khas kata itu "
               "untuk kelas tersebut. Rasio odds = e^skor.")
    a, b = st.columns([3, 1])
    top_n = a.slider("Jumlah fitur teratas", 5, 30, 15)
    jenis = b.radio("Jenis fitur", ["Semua", "Kata (unigram)", "Frasa (bigram+)"])
    inf = influential(300)
    if jenis == "Kata (unigram)":
        inf = inf[~inf["Fitur"].str.contains(" ")]
    elif jenis == "Frasa (bigram+)":
        inf = inf[inf["Fitur"].str.contains(" ")]
    inf = pd.concat([inf[inf["Kelas"] == c].sort_values("Skor Pengaruh", ascending=False).head(top_n) for c in classes])
    cols = st.columns(len(classes))
    for col, c in zip(cols, classes):
        dd = inf[inf["Kelas"] == c].sort_values("Skor Pengaruh")
        fig = px.bar(dd, x="Skor Pengaruh", y="Fitur", orientation="h", title=c, color_discrete_sequence=[cmap[c]],
                     hover_data=["Rasio Odds"])
        fig.update_layout(height=max(380, top_n * 27), margin=dict(l=0, r=0, t=40, b=0))
        col.plotly_chart(fig, width="stretch")

    st.subheader("Heatmap log-probabilitas (kata × kelas)")
    pilih = []
    for i in range(len(classes)):
        pilih += list(np.argsort(flp[i] - np.delete(flp, i, axis=0).mean(axis=0))[::-1][:8])
    pilih = list(dict.fromkeys(pilih))
    hm = pd.DataFrame(flp[:, pilih].T, index=fitur[pilih], columns=classes)
    st.plotly_chart(px.imshow(hm, aspect="auto", color_continuous_scale="RdPu", text_auto=".1f",
                                height=max(420, len(hm) * 22)), width="stretch")

    st.subheader("🔎 Cek pengaruh sebuah kata")
    w = st.text_input("Kata/frasa (hasil stemming, mis. error, mahal, bantu)").strip().lower()
    if w:
        idx = [j for j, f in enumerate(fitur) if w in f]
        if not idx:
            st.info("Kata tidak ada di kosakata model (mungkin terlalu jarang atau berbeda bentuk stemming).")
        else:
            rows = []
            for j in idx[:40]:
                r = {"Fitur": fitur[j]}
                for i, c in enumerate(classes):
                    r[f"Skor {c}"] = flp[i, j] - np.delete(flp[:, j], i).mean()
                r["Condong ke"] = classes[int(np.argmax([r[f"Skor {c}"] for c in classes]))]
                rows.append(r)
            st.dataframe(pd.DataFrame(rows).style.format({f"Skor {c}": "{:.2f}" for c in classes}), width="stretch")
    st.download_button("⬇ Unduh Influential Features (CSV)", influential(30).to_csv(index=False).encode("utf-8"),
                       "influential_features.csv", "text/csv")

# ---------------------------------------------------------------- ASPEK & ANOMALI
with tabs[5]:
    st.subheader("🧩 Analisis Aspek")
    asp = aspect_table(df, len(df))
    a, b = st.columns([3, 2])
    long = asp.melt("Aspek", ["% Negatif", "% Netral", "% Positif"], var_name="Sentimen", value_name="Persen")
    long["Sentimen"] = long["Sentimen"].str.replace("% ", "", regex=False)
    a.plotly_chart(px.bar(long, y="Aspek", x="Persen", color="Sentimen", orientation="h", barmode="stack",
                          color_discrete_map=cmap, title="Komposisi Sentimen per Aspek (%)",
                          category_orders={"Aspek": asp.sort_values("% Negatif")["Aspek"].tolist()}), width="stretch")
    b.plotly_chart(px.bar(asp.sort_values("Jumlah Ulasan"), x="Jumlah Ulasan", y="Aspek", orientation="h",
                          color="Rata-rata Rating", color_continuous_scale="RdYlGn", range_color=[2, 5],
                          title="Seberapa Sering Aspek Disebut"), width="stretch")
    st.dataframe(asp.style.format({"Porsi (%)": "{:.1f}", "Rata-rata Rating": "{:.2f}", "% Negatif": "{:.1f}",
                                   "% Netral": "{:.1f}", "% Positif": "{:.1f}"})
                 .background_gradient(cmap="Reds", subset=["% Negatif"]), width="stretch")

    st.subheader("Telusuri Ulasan per Aspek")
    c1, c2 = st.columns(2)
    ap = c1.selectbox("Aspek", list(ASPEK))
    lp = c2.multiselect("Sentimen", classes, default=[c for c in classes if c == "Negatif"] or classes)
    m = df["content"].astype(str).str.lower().str.contains(ASPEK[ap], regex=True) & df["label"].isin(lp)
    st.write(f"{int(m.sum())} ulasan")
    st.dataframe(df.loc[m, [c for c in ["at", "score", "label", "content"] if c in df.columns]].head(100), width="stretch")

    st.divider()
    st.subheader("⚠ Ulasan Anomali: Teks Berlawanan dengan Rating")
    st.caption("Model yakin sentimen teksnya berbeda dari rating (mis. rating 1 tetapi teks memuji). Berguna untuk menemukan "
               "salah pilih bintang atau ulasan ambigu.")
    conf = st.slider("Minimal keyakinan model", 0.5, 0.99, 0.80, 0.01)
    pa = model.predict_proba(vec.transform(df["clean_content"]))
    an = df[["score", "label", "content"]].copy()
    an["prediksi_teks"] = model.classes_[pa.argmax(axis=1)]; an["keyakinan"] = pa.max(axis=1)
    an = an[(an["prediksi_teks"] != an["label"]) & (an["keyakinan"] >= conf)].sort_values("keyakinan", ascending=False)
    st.write(f"{len(an)} ulasan anomali")
    st.dataframe(an.head(100).style.format({"keyakinan": "{:.2f}"}), width="stretch")

# ---------------------------------------------------------------- PREDIKSI
with tabs[6]:
    st.subheader("🔮 Prediksi Sentimen Real-time")
    ex = st.selectbox("Contoh cepat", ["(ketik sendiri)", "Aplikasinya bagus banget, materi lengkap dan gurunya jelas!",
                                       "Sering error, tidak bisa login, kecewa sekali",
                                       "Lumayan lah tapi masih banyak yang perlu diperbaiki"])
    text = st.text_area("Masukkan ulasan", "" if ex == "(ketik sendiri)" else ex, height=110)
    if st.button("Analisis Sentimen", type="primary") and text.strip():
        with st.spinner("Memproses..."):
            out = prediksi(text)
            simpan_stem_cache()
        if out is None:
            st.warning("Teks kosong setelah dibersihkan. Coba kalimat lain.")
        else:
            emo = {"Positif": "😄", "Netral": "😐", "Negatif": "😡"}.get(out["pred"], "🤔")
            a, b = st.columns([1, 2])
            a.markdown(f'<div class="kpi" style="border-left-color:{cmap[out["pred"]]}"><div class="l">Prediksi</div>'
                       f'<div class="v">{emo} {out["pred"]}</div></div>', unsafe_allow_html=True)
            a.caption(f"Teks bersih: `{out['clean']}`")
            pr = pd.DataFrame({"Kelas": list(out["proba"]), "Probabilitas": list(out["proba"].values())})
            b.plotly_chart(px.bar(pr, x="Probabilitas", y="Kelas", orientation="h", color="Kelas", color_discrete_map=cmap,
                                  range_x=[0, 1], text=pr["Probabilitas"].map("{:.1%}".format)), width="stretch")
            if not out["contrib"].empty:
                st.markdown("**Kata yang paling mendorong keputusan ini:**")
                st.dataframe(out["contrib"], width="stretch")

    st.divider()
    st.subheader("📂 Prediksi Batch (CSV)")
    up = st.file_uploader("CSV dengan kolom 'content'", type="csv", key="batch")
    if up is not None:
        bdf = pd.read_csv(up)
        if "content" in bdf.columns:
            with st.spinner("Memproses ulasan..."):
                cl = bdf["content"].astype(str).apply(bersihkan_teks)
                simpan_stem_cache()
            P = model.predict_proba(vec.transform(cl))
            bdf["prediksi"] = model.classes_[P.argmax(axis=1)]; bdf["keyakinan"] = P.max(axis=1).round(3)
            st.dataframe(bdf.head(100), width="stretch")
            st.plotly_chart(px.pie(bdf, names="prediksi", hole=.5, color="prediksi", color_discrete_map=cmap), width="stretch")
            st.download_button("⬇ Unduh hasil", bdf.to_csv(index=False).encode("utf-8"), "hasil_prediksi.csv", "text/csv")
        else:
            st.error("Kolom 'content' tidak ditemukan.")

# ---------------------------------------------------------------- METODOLOGI
with tabs[7]:
    st.subheader("📘 Alur Proses")
    st.markdown("""
Proses berikut dijalankan di **notebook**; aplikasi ini hanya menampilkan hasilnya.

1. **Load data** ulasan Ruangguru dari Google Play Store.
2. **Pelabelan** dari rating: 1–2 Negatif, 3 Netral, 4–5 Positif.
3. **Pra-pemrosesan**: case folding, hapus URL/simbol, normalisasi slang, stopword (negasi dipertahankan), stemming Sastrawi.
4. **Split stratified** 80/20 (random_state=42), lalu **TF-IDF** dan **SMOTE** hanya pada data latih.
5. **Tuning** GridSearchCV (F1-macro) dan pelatihan **Multinomial Naive Bayes**.
6. **Evaluasi**: Accuracy, Precision, Recall, F1, confusion matrix, ROC-AUC OvR, cross-validation.
7. **Influential Features** dari `feature_log_prob_` dan **analisis aspek**.

Aplikasi membentuk ulang data uji yang sama dengan notebook (split deterministik) lalu hanya memanggil `predict`, sehingga angkanya identik tanpa melatih ulang.

**Catatan:** label berasal dari rating, jadi model mengukur kemampuan menebak rating dari teks, bukan sentimen yang dianotasi manual.
""")
    st.download_button("⬇ Unduh data bersih + label (CSV)", df.to_csv(index=False).encode("utf-8"), "data_bersih.csv", "text/csv")