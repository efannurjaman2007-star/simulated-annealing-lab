import math
import random
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Simulated Annealing Lab", page_icon="🔥", layout="wide")
st.title("Simulated Annealing Lab")
st.caption("Aplikasi web interaktif untuk optimasi menggunakan algoritma Simulated Annealing.")
st.markdown("**Tujuan:** mencari nilai minimum fungsi tujuan pada rentang x tertentu.")

FUNCTIONS = {
    "f(x) = x² + 10 sin(x)": lambda x: x*x + 10*math.sin(x),
    "f(x) = x² + 5 sin(3x)": lambda x: x*x + 5*math.sin(3*x),
}

def simulated_annealing(func, xmin, xmax, t0, tmin, cooling, max_iter, step, seed):
    rng = random.Random(seed)
    x = rng.uniform(xmin, xmax)
    value = func(x)
    best_x, best_value = x, value
    temp = float(t0)
    history = [{"iterasi": 0, "x": x, "nilai": value, "suhu": temp, "solusi_terbaik": best_value}]
    for i in range(1, max_iter + 1):
        if temp <= tmin:
            break
        candidate = max(xmin, min(xmax, x + rng.uniform(-step, step)))
        candidate_value = func(candidate)
        delta = candidate_value - value
        if delta <= 0 or rng.random() < math.exp(-delta / temp):
            x, value = candidate, candidate_value
        if value < best_value:
            best_x, best_value = x, value
        history.append({"iterasi": i, "x": x, "nilai": value, "suhu": temp, "solusi_terbaik": best_value})
        temp *= cooling
    return best_x, best_value, history

with st.sidebar:
    st.header("⚙️ Pengaturan simulasi")
    function_name = st.selectbox("Fungsi tujuan", list(FUNCTIONS))
    seed = st.number_input("Seed acak", 0, 999999, 42, 1)
    st.subheader("Rentang pencarian")
    xmin = st.number_input("Batas bawah x", value=-10.0, step=1.0)
    xmax = st.number_input("Batas atas x", value=10.0, step=1.0)
    st.subheader("Parameter algoritma")
    t0 = st.number_input("Suhu awal", min_value=0.001, value=100.0, step=10.0)
    tmin = st.number_input("Suhu minimum", min_value=0.000001, value=0.1, format="%.6f")
    cooling = st.slider("Laju pendinginan", 0.80, 0.999, 0.95, 0.001)
    max_iter = st.number_input("Maksimum iterasi", 1, 100000, 500, 50)
    step = st.number_input("Ukuran langkah", min_value=0.001, value=1.0, step=0.1)
    show_table = st.checkbox("Tampilkan tabel langkah", True)
    run = st.button("▶ Jalankan simulasi", type="primary", use_container_width=True)

if xmin >= xmax:
    st.error("Batas bawah x harus lebih kecil daripada batas atas x.")
    st.stop()
if tmin >= t0:
    st.error("Suhu minimum harus lebih kecil daripada suhu awal.")
    st.stop()

if run or "sa_result" not in st.session_state:
    bx, bv, history = simulated_annealing(FUNCTIONS[function_name], float(xmin), float(xmax),
        float(t0), float(tmin), float(cooling), int(max_iter), float(step), int(seed))
    st.session_state.sa_result = {"best_x": bx, "best_value": bv, "rows": history,
                                  "function": function_name, "seed": int(seed)}

result = st.session_state.get("sa_result")
rows = result.get("rows") if isinstance(result, dict) else None
if not isinstance(rows, list) or not rows:
    st.warning("Data simulasi belum tersedia. Klik Jalankan simulasi.")
    st.stop()

df = pd.DataFrame(rows)
st.success("Simulasi berhasil dijalankan.")
st.caption(f"Hasil terakhir: {result['function']} | Seed: {result['seed']}")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Jumlah iterasi", f"{len(rows)-1:,}")
c2.metric("Solusi terbaik x", f"{result['best_x']:.5f}")
c3.metric("Nilai minimum ditemukan", f"{result['best_value']:.5f}")
c4.metric("Suhu akhir", f"{rows[-1]['suhu']:.5f}")

tab1,tab2,tab3 = st.tabs(["📈 Grafik pencarian","🌡️ Suhu & nilai","🧠 Cara kerja"])
with tab1:
    fig,ax=plt.subplots(figsize=(10,4))
    func=FUNCTIONS[result["function"]]
    curve_x=[xmin+i*(xmax-xmin)/400 for i in range(401)]
    ax.plot(curve_x,[func(x) for x in curve_x],label="Fungsi tujuan")
    ax.scatter(df["x"],df["nilai"],s=12,alpha=.4,label="Posisi dikunjungi")
    ax.scatter([result["best_x"]],[result["best_value"]],marker="*",s=180,label="Solusi terbaik")
    ax.set_title("Pencarian solusi minimum"); ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.grid(True,alpha=.25); ax.legend()
    st.pyplot(fig,use_container_width=True); plt.close(fig)
with tab2:
    fig,ax1=plt.subplots(figsize=(10,4))
    ax1.plot(df["iterasi"],df["nilai"],label="Nilai solusi")
    ax1.plot(df["iterasi"],df["solusi_terbaik"],"--",label="Nilai terbaik")
    ax1.set_xlabel("Iterasi"); ax1.set_ylabel("f(x)"); ax1.grid(True,alpha=.25)
    ax2=ax1.twinx(); ax2.plot(df["iterasi"],df["suhu"],alpha=.65,label="Suhu"); ax2.set_ylabel("Suhu")
    l1,n1=ax1.get_legend_handles_labels(); l2,n2=ax2.get_legend_handles_labels()
    ax1.legend(l1+l2,n1+n2,loc="best"); ax1.set_title("Perubahan nilai solusi dan suhu")
    st.pyplot(fig,use_container_width=True); plt.close(fig)
with tab3:
    st.markdown("""
    1. **Inisialisasi:** pilih solusi awal secara acak dan tentukan suhu awal.
    2. **Membuat tetangga:** buat kandidat solusi di sekitar solusi saat ini.
    3. **Evaluasi:** hitung nilai fungsi tujuan kandidat.
    4. **Penerimaan:** solusi lebih baik diterima; solusi lebih buruk mungkin diterima dengan probabilitas `exp(-selisih / suhu)`.
    5. **Pendinginan:** turunkan suhu sesuai laju pendinginan.
    6. **Berhenti:** saat suhu minimum tercapai atau iterasi maksimum selesai.
    """)
    st.info("Simulated Annealing adalah metode heuristik. Hasilnya bukan jaminan optimum global.")
if show_table:
    with st.expander("📋 Lihat tabel langkah algoritma"):
        view=df.copy()
        for col in ["x","nilai","suhu","solusi_terbaik"]:
            view[col]=view[col].round(6)
        st.dataframe(view,use_container_width=True,hide_index=True)
st.download_button("⬇️ Unduh riwayat simulasi (CSV)",df.to_csv(index=False).encode("utf-8"),
    "riwayat_simulated_annealing.csv","text/csv")
st.divider()
st.caption("Simulated Annealing Lab • Python + Streamlit")
