import streamlit as st
import database as db
import pandas as pd

def show():
    db.init_db()
    st.title("📂 Database Laporan")
    
    # Refresh data setiap kali halaman dibuka
    df = db.get_all_laporan_as_df()
    
    if not df.empty and 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # --- Metrics Section ---
    if not df.empty:
        total, critical = db.get_summary_stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Laporan", total)
        c2.metric("Status Critical", critical)
        
        try:
            latest = df.iloc[0]['timestamp'].strftime("%d %b %H:%M")
        except:
            latest = "-"
        c3.metric("Update Terakhir", latest)
    
    st.divider()

    # --- Table Section ---
    if not df.empty:
        # Pindahkan Filter ke Expander agar rapi
        with st.expander("🔎 Filter Data", expanded=False):
            status_opts = df['status'].unique().tolist()
            sel_status = st.multiselect("Status:", status_opts, default=status_opts)
            
            gedung_opts = df['gedung'].unique().tolist()
            sel_gedung = st.multiselect("Gedung:", gedung_opts, default=gedung_opts)

        # Logic Filter
        mask = df['status'].isin(sel_status) & df['gedung'].isin(sel_gedung)
        df_show = df[mask]

        st.dataframe(
            df_show,
            width='stretch',
            hide_index=True,
            column_config={
                "id": st.column_config.NumberColumn("ID", width="small"),
                "timestamp": st.column_config.DatetimeColumn("Waktu", format="DD/MM HH:mm"),
                "confidence_score": st.column_config.ProgressColumn("Skor", format="%d%%", min_value=0, max_value=100),
                "jenis_kerusakan": "Detail Temuan"
            }
        )
    else:
        st.info("Belum ada data laporan. Silakan upload video di menu Scanner.")