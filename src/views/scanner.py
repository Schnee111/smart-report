import streamlit as st
import database as db
import utils 
import cv2
import tempfile
import time
import threading
from collections import Counter
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# --- LOGIKA REAL-TIME (WEBRTC) ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.skip_rate = 30
        self.last_frame = None
        self.latest_predictions = []
        self.lock = threading.Lock()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # Frame Skipping untuk performa
        if self.frame_count % self.skip_rate != 0 and self.last_frame is not None:
            return self.last_frame
            
        h, w = img.shape[:2]
        new_h = int(h * (640 / w))
        img_small = cv2.resize(img, (640, new_h))
        
        # Panggil AI
        annotated_img, preds = utils.run_ai_workflow(img_small)
        
        with self.lock:
            self.latest_predictions = preds
            
        self.last_frame = annotated_img
        return annotated_img

# --- LOGIKA PERHITUNGAN SKOR ---
def calculate_score(unique_counts):
    deduction = 0
    is_critical_failure = False
    
    # Kriteria Penilaian 3 Kelas
    if unique_counts.get("dudukan_rusak", 0) > 0:
        is_critical_failure = True
        deduction = 90
    elif unique_counts.get("tanpa_meja", 0) > 0:
        is_critical_failure = True
        deduction = 70

    if not is_critical_failure:
        sobek_count = unique_counts.get("sobek", 0)
        deduction += sobek_count * 15
    
    deduction = min(100, deduction)
    final_score = max(0, 100 - deduction)
    
    if is_critical_failure or final_score < 50:
        status = "Rusak Berat 🛑"
    elif final_score < 85:
        status = "Perlu Perbaikan ⚠️"
    else:
        status = "Layak Pakai ✅"
    
    return final_score, deduction, status

# --- MAIN UI ---
def show():
    db.init_db()
    st.title("📹 AI Facility Audit")
    
    # Metadata Lokasi
    with st.container():
        c1, c2 = st.columns(2)
        lokasi_gedung = c1.selectbox("Gedung", ["FPMIPA A", "FPMIPA B", "FPMIPA C"])
        lokasi_ruang = c2.text_input("Ruangan", placeholder="Contoh: S-304")

    st.divider()
    mode = st.radio("Metode Input:", ["Live Camera (15s Audit)", "Upload Video File"], horizontal=True)

    # ==========================================
    # MODE 1: LIVE CAMERA (AUTO STOP 15S)
    # ==========================================
    if mode == "Live Camera (15s Audit)":
        
        # State Management untuk Live Scan
        if "scan_active" not in st.session_state: st.session_state.scan_active = False
        if "scan_finished" not in st.session_state: st.session_state.scan_finished = False
        if "live_results" not in st.session_state: st.session_state.live_results = Counter()
        if "start_time" not in st.session_state: st.session_state.start_time = 0

        col_vid, col_instr = st.columns([1.8, 1])
        
        with col_vid:
            # Webrtc selalu mount agar kamera siap
            ctx = webrtc_streamer(
                key="scanner-live", 
                video_processor_factory=VideoProcessor,
                mode=WebRtcMode.SENDRECV,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False},
            )

        with col_instr:
            st.markdown("##### 📋 Instruksi Audit")
            st.info("1. Tekan tombol **MULAI SCAN**.\n2. Putari objek 360° perlahan.\n3. Waktu scan otomatis berhenti dalam **15 detik**.")
            
            # Tombol Kontrol
            if not st.session_state.scan_active and not st.session_state.scan_finished:
                if st.button("▶️ MULAI SCAN (15 Detik)", type="primary", use_container_width=True):
                    st.session_state.scan_active = True
                    st.session_state.live_results = Counter() # Reset hasil
                    st.session_state.start_time = time.time()
                    st.rerun()

            # Tampilan Saat Scanning Berjalan
            if st.session_state.scan_active:
                elapsed = time.time() - st.session_state.start_time
                remaining = 15 - elapsed
                
                # Progress Bar Waktu
                prog_val = min(elapsed / 15, 1.0)
                st.progress(prog_val, text=f"⏳ Menganalisis... Sisa waktu: {int(remaining)}s")
                
                # Ambil data dari processor
                if ctx.video_transformer:
                    with ctx.video_transformer.lock:
                        preds = ctx.video_transformer.latest_predictions
                    
                    if preds:
                        curr = Counter([p['class'] for p in preds])
                        # Max Pooling Aggregation
                        for k, v in curr.items():
                            if v > st.session_state.live_results[k]:
                                st.session_state.live_results[k] = v

                # Cek jika waktu habis
                if elapsed >= 15:
                    st.session_state.scan_active = False
                    st.session_state.scan_finished = True
                    st.rerun()
                else:
                    time.sleep(0.5) # Refresh rate UI
                    st.rerun() # Force UI update untuk timer berjalan halus

            # Tampilan Setelah Selesai
            if st.session_state.scan_finished:
                st.success("✅ Waktu Habis! Analisis Selesai.")
                
                final_res = st.session_state.live_results
                score, deduc, stat = calculate_score(final_res)
                
                st.metric("Skor Akhir", f"{score}%", f"-{deduc}%", delta_color="inverse")
                st.metric("Status", stat)
                st.json(dict(final_res))
                
                if st.button("💾 Simpan Hasil", type="primary", use_container_width=True):
                    if lokasi_ruang:
                        db.create_laporan(lokasi_gedung, lokasi_ruang, str(dict(final_res)), score, stat, "Live Audit (15s)")
                        st.success("Tersimpan!")
                        # Reset untuk scan berikutnya
                        time.sleep(2)
                        st.session_state.scan_finished = False
                        st.rerun()
                    else:
                        st.error("Isi Ruangan dulu!")
                
                if st.button("🔄 Ulangi Scan"):
                    st.session_state.scan_finished = False
                    st.rerun()

    # ==========================================
    # MODE 2: UPLOAD VIDEO (CACHING SYSTEM)
    # ==========================================
    elif mode == "Upload Video File":
        uploaded_video = st.file_uploader("Pilih video (.mp4)", type=["mp4", "avi"])
        
        # State untuk caching hasil upload agar tidak reprocessing
        if "last_video_name" not in st.session_state: st.session_state.last_video_name = None
        if "video_results" not in st.session_state: st.session_state.video_results = None

        if uploaded_video:
            # Cek apakah ini video baru atau video yang sama?
            is_new_video = (st.session_state.last_video_name != uploaded_video.name)
            
            if is_new_video:
                # --- PROSES VIDEO (Hanya Jalan Sekali) ---
                if not lokasi_ruang:
                    st.error("⚠️ Mohon isi Nama Ruangan di atas terlebih dahulu!")
                    st.stop()

                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(uploaded_video.read())
                vf = cv2.VideoCapture(tfile.name)
                
                st.write("---")
                col_video, col_prog = st.columns([1.8, 1])
                
                with col_prog:
                    st.info("⚙️ Sedang Menganalisis Video...")
                    prog_bar = st.progress(0)
                    txt_stat = st.empty()
                
                with col_video:
                    stframe = st.empty()

                video_defects = Counter()
                total_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
                SKIP = 30
                curr = 0
                
                while vf.isOpened():
                    ret, frame = vf.read()
                    if not ret: break
                    curr += 1
                    
                    if curr % 5 == 0: prog_bar.progress(min(curr/total_frames, 1.0))
                    if curr % SKIP != 0: continue
                    
                    h, w = frame.shape[:2]
                    new_h = int(h * (480 / w))
                    frame_small = cv2.resize(frame, (480, new_h))
                    
                    annotated, preds = utils.run_ai_workflow(frame_small)
                    
                    # Aggregation
                    frame_c = Counter([p['class'] for p in preds])
                    for k, v in frame_c.items():
                        if v > video_defects[k]: video_defects[k] = v
                    
                    stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

                vf.release()
                
                # SIMPAN HASIL KE CACHE SESSION STATE
                st.session_state.last_video_name = uploaded_video.name
                st.session_state.video_results = video_defects
                st.rerun() # Rerun agar masuk ke blok 'else' (menampilkan hasil)

            else:
                # --- TAMPILKAN HASIL DARI CACHE (Tidak Proses Ulang) ---
                res = st.session_state.video_results
                score, deduc, stat = calculate_score(res)
                
                st.success("Analisis Video Selesai (Cached)")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Temuan Unik", sum(res.values()))
                c2.metric("Skor", f"{score}%", f"-{deduc}%", delta_color="inverse")
                c3.metric("Status", stat)
                
                st.json(dict(res))
                
                if st.button("💾 Simpan Laporan Video", type="primary", use_container_width=True):
                    db.create_laporan(lokasi_gedung, lokasi_ruang, str(dict(res)), score, stat, f"Video: {uploaded_video.name}")
                    st.success("Data tersimpan ke database!")