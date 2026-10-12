import streamlit as st
import pandas as pd
import statsmodels.api as sm
import plotly.express as px

st.set_page_config(page_title="Interactive Econometric Lab - DataAksi", layout="wide")

st.title("📊 Interactive Econometric Lab (Version 2.0)")
st.markdown("**DataAksi • Econometric Analysis Laboratory (Prodi Bisnis Digital FEB UNM)**")
st.write("---")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Alam-A99/DATAAKSI/main/dataset_fiskal_97_hipoFull.csv"
    return pd.read_csv(url)

try:
    df = load_data()
    menu = st.sidebar.selectbox("Pilih Menu Lab", ["Input & Pilih Data", "Model & OLS Regression", "Visualisasi 3D"])
    
    if menu == "Input & Pilih Data":
        st.subheader("📁 Preview Dataset Fiskal Daerah")
        st.dataframe(df.head(15), use_container_width=True)
        st.info(f"Total Baris Data: {df.shape[0]} | Total Kolom: {df.shape[1]}")
        
    elif menu == "Model & OLS Regression":
        st.subheader("⚙️ Estimasi Model Regresi Berganda (OLS)")
        daftar_kota = df['Kota'].unique().tolist()
        pilih_kota = st.selectbox("Pilih Wilayah / Kota untuk Dianalisis:", daftar_kota)
        
        df_filtered = df[df['Kota'] == pilih_kota]
        st.write(f"Menampilkan analisis untuk: **{pilih_kota}** (Jumlah sampel: {len(df_filtered)} tahun)")
        
        if len(df_filtered) > 2:
            X = df_filtered[['bm_pendidikan', 'PAD']]
            y = df_filtered['DAU_Earmark']
            X = sm.add_constant(X)
            model = sm.OLS(y, X)
            results = model.fit()
            
            st.text(str(results.summary()))
            
            col1, col2, col3 = st.columns(3)
            col1.metric("R-Squared ($R^2$)", f"{results.rsquared:.4f}")
            col2.metric("F-Statistic", f"{results.fvalue:.4f}")
            col3.metric("Prob (F-Statistic)", f"{results.f_pvalue:.4e}")
        else:
            st.warning("Data untuk kota ini terlalu sedikit untuk dilakukan regresi.")
            
    elif menu == "Visualisasi 3D":
        st.subheader("🌐 Visualisasi Grafik 3D Interaktif")
        pilih_kota_vis = st.selectbox("Pilih Wilayah untuk Grafik:", df['Kota'].unique().tolist())
        df_vis = df[df['Kota'] == pilih_kota_vis]
        
        fig = px.scatter_3d(
            df_vis,
            x='bm_pendidikan',
            y='PAD',
            z='DAU_Earmark',
            color='DAU_Earmark',
            color_continuous_scale='Plasma',
            title=f'Visualisasi 3D Fiskal - {pilih_kota_vis}'
        )
        fig.update_layout(scene=dict(
            xaxis_title='Belanja Pendidikan',
            yaxis_title='PAD',
            zaxis_title='DAU Earmark'
        ))
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Gagal memuat data: {e}")
