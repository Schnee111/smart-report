import streamlit as st
import database as db
import utils 
import cv2
import tempfile
import time
import os
from collections import Counter
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# --- CONFIG ---
RECORD_TIME = 15 # Detik

# --- LOGIKA PEREKAMAN (TANPA AI - SUPAYA LANCAR) ---
class RecorderProcessor(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.out = None
        self.temp_filename = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        self.is_recording = True
        self.start_time = time.time()

    def transform(self, frame):
        # Konversi format WebRTC ke OpenCV
        img = frame.to_ndarray(format="bgr24")
        
        # Inisialisasi VideoWriter pada frame pertama
        if self.out is None:
            h, w = img.shape[:2]
            # Gunakan 'mp4v' agar kompatibel
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(self.temp_filename, fourcc, 20.0, (w, h))
        
        # Tulis frame ke file (Sangat Cepat, Tidak Lag)
        if self.is_recording:
            self.out.write(img)
            self.frame_count += 1
            
        return img # Kembalikan gambar asli agar user melihat dirinya (Mirror)

    def stop_recording(self):
        self.is_recording = False
        if self.out:
            self.out.release()
            self.out = None

# --- LOGIKA SKOR ---
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

# --- UI UTAMA ---
def show():
    db.init_db()
    st.title("📹 AI Facility Audit")
    
    with st.container():
        c1, c2 = st.columns(2)
        lokasi_gedung = c1.selectbox("Gedung", ["FPMIPA A", "FPMIPA B", "FPMIPA C"])
        lokasi_ruang = c2.text_input("Ruangan", placeholder="Contoh: S-304")

    st.divider()
    mode = st.radio("Metode Input:", ["Kamera HP (Rekam -> Proses)", "Upload Video File"], horizontal=True)

    # ==========================================
    # MODE 1: REKAM DULU -> BARU PROSES
    # ==========================================
    if mode == "Kamera HP (Rekam -> Proses)":
        
        # State Management
        if "phase" not in st.session_state: st.session_state.phase = "IDLE" # IDLE, RECORDING, PROCESSING, DONE
        if "recorded_file" not in st.session_state: st.session_state.recorded_file = None
        if "start_rec_time" not in st.session_state: st.session_state.start_rec_time = 0

        # 1. FASE IDLE / RECORDING
        if st.session_state.phase in ["IDLE", "RECORDING"]:
            col_cam, col_info = st.columns([1.8, 1])
            
            with col_cam:
                # WebRTC Streamer
                ctx = webrtc_streamer(
                    key="scanner-recorder", 
                    video_processor_factory=RecorderProcessor,
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                    media_stream_constraints={
                        "video": {
                            "width": {"ideal": 640}, # Tetap ringan
                            "height": {"ideal": 480},
                            "facingMode": "environment"
                        },
                        "audio": False
                    },
                    async_processing=True
                )

            with col_info:
                st.markdown("##### 📸 Instruksi")
                st.info(f"1. Arahkan kamera.\n2. Klik **MULAI REKAM**.\n3. Tunggu {RECORD_TIME} detik.\n4. AI akan memproses video setelah selesai.")
                
                # Logic Tombol Start
                if st.session_state.phase == "IDLE":
                    if st.button("🔴 MULAI REKAM", type="primary", width='stretch'):
                        if not lokasi_ruang:
                            st.error("Isi Ruangan Dulu!")
                        else:
                            st.session_state.phase = "RECORDING"
                            st.session_state.start_rec_time = time.time()
                            st.rerun()

                # Logic Timer Saat Recording
                if st.session_state.phase == "RECORDING":
                    elapsed = time.time() - st.session_state.start_rec_time
                    remaining = RECORD_TIME - elapsed
                    
                    st.progress(min(elapsed / RECORD_TIME, 1.0), text=f"🎥 Merekam... {int(remaining)}s")
                    
                    # Cek Waktu Habis
                    if elapsed >= RECORD_TIME:
                        # Ambil file dari processor
                        if ctx.video_transformer:
                            ctx.video_transformer.stop_recording()
                            st.session_state.recorded_file = ctx.video_transformer.temp_filename
                        
                        st.session_state.phase = "PROCESSING"
                        st.rerun()
                    else:
                        time.sleep(0.5)
                        st.rerun()

        # 2. FASE PROCESSING (AI BEKERJA DI SINI)
        elif st.session_state.phase == "PROCESSING":
            st.info("⚙️ Video tersimpan! Sekarang sedang memindai kerusakan (AI Processing)...")
            
            video_path = st.session_state.recorded_file
            if not video_path or not os.path.exists(video_path):
                st.error("Gagal menyimpan video. Coba lagi.")
                st.session_state.phase = "IDLE"
                st.stop()

            # UI Progress
            c_vid, c_res = st.columns([1.8, 1])
            with c_vid: stframe = st.empty()
            with c_res: 
                prog_bar = st.progress(0)
                txt_stat = st.empty()
                live_json = st.empty()

            # Loop Processing (Sama seperti Upload Video)
            vf = cv2.VideoCapture(video_path)
            total_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
            video_defects = Counter()
            curr = 0
            
            while vf.isOpened():
                ret, frame = vf.read()
                if not ret: break
                curr += 1
                
                # Update UI
                if curr % 5 == 0: 
                    prog_bar.progress(min(curr/total_frames, 1.0))
                    txt_stat.caption(f"Analyzing Frame: {curr}/{total_frames}")
                
                # Skip Frame (Biar agak cepat, tapi tetap detil)
                if curr % 15 != 0: continue 

                # Resize & AI
                h, w = frame.shape[:2]
                new_h = int(h * (480 / w))
                frame_small = cv2.resize(frame, (480, new_h))
                
                annotated, preds = utils.run_ai_workflow(frame_small)
                
                # Aggregasi
                frame_c = Counter([p['class'] for p in preds])
                for k, v in frame_c.items():
                    if v > video_defects[k]: video_defects[k] = v
                
                # Tampilkan Bounding Box
                stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width='stretch')
                live_json.json(dict(video_defects))

            vf.release()
            
            # Auto Save
            score, deduc, stat = calculate_score(video_defects)
            db.create_laporan(lokasi_gedung, lokasi_ruang, str(dict(video_defects)), score, stat, "Live-Rec Audit")
            
            # Pindah ke Fase Selesai
            st.session_state.final_results = video_defects
            st.session_state.phase = "DONE"
            st.rerun()

        # 3. FASE DONE (TAMPIL HASIL)
        elif st.session_state.phase == "DONE":
            st.balloons()
            st.success(f"✅ Analisis Selesai! Data Ruangan {lokasi_ruang} tersimpan.")
            
            res = st.session_state.final_results
            score, deduc, stat = calculate_score(res)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Temuan", sum(res.values()))
            c2.metric("Skor", f"{score}%", f"-{deduc}%", delta_color="inverse")
            c3.metric("Status", stat)
            
            st.json(dict(res))
            
            if st.button("🔄 Audit Ruangan Lain"):
                st.session_state.phase = "IDLE"
                st.session_state.recorded_file = None
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
                    
                    stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width='stretch')

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
                    st.success(f"✅ Analisis Selesai. Data Kursi di {lokasi_ruang} Tersimpan Otomatis!")
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