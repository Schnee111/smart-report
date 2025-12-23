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
        
        if self.frame_count % self.skip_rate != 0 and self.last_frame is not None:
            return self.last_frame
            
        h, w = img.shape[:2]
        new_h = int(h * (640 / w))
        img_small = cv2.resize(img, (640, new_h))
        
        annotated_img, preds = utils.run_ai_workflow(img_small)
        
        with self.lock:
            self.latest_predictions = preds
            
        self.last_frame = annotated_img
        return annotated_img

# --- LOGIKA PERHITUNGAN SKOR ---
def calculate_score(unique_counts):
    deduction = 0
    is_critical_failure = False
    
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
    
    with st.container():
        c1, c2 = st.columns(2)
        lokasi_gedung = c1.selectbox("Gedung", ["FPMIPA A", "FPMIPA B", "FPMIPA C"])
        lokasi_ruang = c2.text_input("Ruangan", placeholder="Contoh: S-304")

    st.divider()
    mode = st.radio("Metode Input:", ["Live Camera (15s Audit)", "Upload Video File"], horizontal=True)

    # ==========================================
    # MODE 1: LIVE CAMERA (AUTO SAVE)
    # ==========================================
    if mode == "Live Camera (15s Audit)":
        
        if "scan_active" not in st.session_state: st.session_state.scan_active = False
        if "scan_finished" not in st.session_state: st.session_state.scan_finished = False
        if "live_results" not in st.session_state: st.session_state.live_results = Counter()
        if "start_time" not in st.session_state: st.session_state.start_time = 0
        if "save_status" not in st.session_state: st.session_state.save_status = None # Untuk pesan sukses

        col_vid, col_instr = st.columns([1.8, 1])
        
        with col_vid:
            ctx = webrtc_streamer(
                key="scanner-live", 
                video_processor_factory=VideoProcessor,
                mode=WebRtcMode.SENDRECV,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False},
            )

        with col_instr:
            st.markdown("##### 📋 Instruksi")
            st.info("1. Klik **MULAI SCAN**.\n2. Putari objek 360°.\n3. **Otomatis** berhenti & simpan dalam 15 detik.")
            
            # Tombol Mulai
            if not st.session_state.scan_active and not st.session_state.scan_finished:
                if st.button("▶️ MULAI SCAN (15 Detik)", type="primary", use_container_width=True):
                    if not lokasi_ruang:
                        st.error("⚠️ Isi Ruangan dulu!")
                    else:
                        st.session_state.scan_active = True
                        st.session_state.live_results = Counter()
                        st.session_state.start_time = time.time()
                        st.session_state.save_status = None
                        st.rerun()

            # Proses Scan Berjalan
            if st.session_state.scan_active:
                elapsed = time.time() - st.session_state.start_time
                remaining = 15 - elapsed
                
                prog_val = min(elapsed / 15, 1.0)
                st.progress(prog_val, text=f"⏳ Menganalisis... {int(remaining)}s")
                
                if ctx.video_transformer:
                    with ctx.video_transformer.lock:
                        preds = ctx.video_transformer.latest_predictions
                    if preds:
                        curr = Counter([p['class'] for p in preds])
                        for k, v in curr.items():
                            if v > st.session_state.live_results[k]:
                                st.session_state.live_results[k] = v

                # JIKA WAKTU HABIS -> AUTO SAVE
                if elapsed >= 15:
                    final_res = st.session_state.live_results
                    score, deduc, stat = calculate_score(final_res)
                    
                    # Simpan ke DB Langsung
                    db.create_laporan(lokasi_gedung, lokasi_ruang, str(dict(final_res)), score, stat, "Auto-Save Live (15s)")
                    
                    # Update State
                    st.session_state.scan_active = False
                    st.session_state.scan_finished = True
                    st.session_state.save_status = "Sukses"
                    st.rerun() # Refresh agar masuk ke blok finish
                else:
                    time.sleep(0.5)
                    st.rerun()

            # Tampilan Setelah Selesai
            if st.session_state.scan_finished:
                if st.session_state.save_status == "Sukses":
                    st.balloons()
                    st.success(f"✅ Waktu Habis. Data Ruangan {lokasi_ruang} tersimpan otomatis!")
                
                res = st.session_state.live_results
                score, deduc, stat = calculate_score(res)
                
                st.metric("Skor Akhir", f"{score}%", f"-{deduc}%", delta_color="inverse")
                st.metric("Status", stat)
                st.json(dict(res))
                
                if st.button("🔄 Scan Objek Lain"):
                    st.session_state.scan_finished = False
                    st.session_state.save_status = None
                    st.rerun()

    # ==========================================
    # MODE 2: UPLOAD VIDEO (AUTO SAVE NO LOOP)
    # ==========================================
    elif mode == "Upload Video File":
        uploaded_video = st.file_uploader("Pilih video (.mp4)", type=["mp4", "avi"])
        
        # State Management
        if "last_video_name" not in st.session_state: st.session_state.last_video_name = None
        if "video_results" not in st.session_state: st.session_state.video_results = None
        if "upload_success" not in st.session_state: st.session_state.upload_success = False

        if uploaded_video:
            # Cek apakah video ini BARU?
            is_new_video = (st.session_state.last_video_name != uploaded_video.name)
            
            if is_new_video:
                # --- BLOK PROSES (Hanya jalan 1x per file) ---
                if not lokasi_ruang:
                    st.error("⚠️ Mohon isi Nama Ruangan di atas terlebih dahulu!")
                    st.stop()

                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(uploaded_video.read())
                vf = cv2.VideoCapture(tfile.name)
                
                st.write("---")
                col_video, col_prog = st.columns([1.8, 1])
                
                with col_prog:
                    st.info("⚙️ Menganalisis Video & Auto-Save...")
                    prog_bar = st.progress(0)
                
                with col_video:
                    stframe = st.empty()

                video_defects = Counter()
                total_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
                curr = 0
                
                while vf.isOpened():
                    ret, frame = vf.read()
                    if not ret: break
                    curr += 1
                    
                    if curr % 5 == 0: prog_bar.progress(min(curr/total_frames, 1.0))
                    if curr % 30 != 0: continue # Skip frame
                    
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
                
                # --- AUTO SAVE LOGIC (Di sini kuncinya) ---
                final_score, deduction, status = calculate_score(video_defects)
                
                # 1. Simpan DB
                db.create_laporan(lokasi_gedung, lokasi_ruang, str(dict(video_defects)), final_score, status, f"Auto-Video: {uploaded_video.name}")
                
                # 2. Update Session State (Agar tidak looping)
                st.session_state.last_video_name = uploaded_video.name
                st.session_state.video_results = video_defects
                st.session_state.upload_success = True
                
                # 3. Rerun untuk refresh UI ke mode "Tampil Hasil"
                st.rerun()

            else:
                # --- BLOK TAMPIL HASIL (Statik) ---
                # Masuk sini jika nama video sama dengan yang di memori
                
                if st.session_state.upload_success:
                    st.balloons()
                    st.success(f"✅ Analisis Selesai. Data Ruangan {lokasi_ruang} Tersimpan Otomatis!")
                    # Reset flag success agar balloon tidak muncul terus jika user klik tab lain
                    st.session_state.upload_success = False 
                else:
                    st.info("📂 Menampilkan data hasil analisis sebelumnya.")

                res = st.session_state.video_results
                score, deduc, stat = calculate_score(res)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Temuan Unik", sum(res.values()))
                c2.metric("Skor", f"{score}%", f"-{deduc}%", delta_color="inverse")
                c3.metric("Status", stat)
                
                st.json(dict(res))
                st.caption("ℹ️ Untuk memproses video lain, silakan klik 'Browse files' dan pilih file baru.")