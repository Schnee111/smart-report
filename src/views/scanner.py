import streamlit as st
import database as db
import utils 
import cv2
import tempfile
import time
import threading
from collections import Counter
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- LOGIKA REAL-TIME ---
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

def calculate_score(unique_counts):
    deduction = 0
    deduction += unique_counts.get("Retak", 0) * 15
    deduction += unique_counts.get("Patah", 0) * 20
    deduction += unique_counts.get("Bocor", 0) * 10
    deduction += unique_counts.get("Noda", 0) * 2
    deduction += unique_counts.get("Goresan", 0) * 2
    
    final_score = max(0, 100 - deduction)
    status = "Good"
    if final_score < 60: status = "Critical"
    elif final_score < 85: status = "Minor"
    
    return final_score, deduction, status

def show():
    db.init_db()
    st.title("📹 AI Facility Audit")
    
    with st.container():
        c1, c2 = st.columns(2)
        lokasi_gedung = c1.selectbox("Gedung", ["FPMIPA A", "FPMIPA B", "Lab Fisika"])
        lokasi_ruang = c2.text_input("Ruangan", placeholder="Contoh: R. 304")

    st.divider()
    mode = st.radio("Metode Input:", ["Live Camera", "Upload Video"], horizontal=True)

    # MODE 1: LIVE (Tetap Manual karena stream tak berujung)
    if mode == "Live Camera":
        # ... (Kode Live Camera Sama Seperti Sebelumnya) ...
        # (Copy bagian Live Camera dari kode sebelumnya jika perlu, atau biarkan fokus ke Upload)
        st.info("Fitur Live Camera memerlukan tombol simpan manual karena streaming berjalan terus menerus.")
        col_vid, col_stat = st.columns([1.8, 1])
        with col_vid:
            ctx = webrtc_streamer(key="scanner-live", video_processor_factory=VideoProcessor)
        # ... (Sisa logika live) ...

    # MODE 2: UPLOAD VIDEO (OTOMATIS)
    elif mode == "Upload Video":
        uploaded_video = st.file_uploader("Pilih video (.mp4)", type=["mp4", "avi"])
        
        if uploaded_video:
            # Cek apakah lokasi sudah diisi?
            if not lokasi_ruang:
                st.error("⚠️ Mohon isi Nama Ruangan di atas terlebih dahulu sebelum upload video!")
                st.stop()

            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_video.read())
            vf = cv2.VideoCapture(tfile.name)
            
            total_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
            st.write("---")
            
            col_video, col_progress = st.columns([1.8, 1]) 
            
            with col_progress:
                st.markdown("##### ⏳ Status Audit")
                prog_bar = st.progress(0)
                txt_status = st.empty()
                live_findings = st.empty()
            
            with col_video:
                stframe = st.empty()

            SKIP_FRAMES = 30 
            curr_frame = 0
            video_defects = Counter()
            
            # Loop Processing
            while vf.isOpened():
                ret, frame = vf.read()
                if not ret: break
                
                curr_frame += 1
                if curr_frame % 5 == 0: 
                    prog_bar.progress(min(curr_frame / total_frames, 1.0))
                    txt_status.caption(f"Processing Frame: {curr_frame}/{total_frames}")

                if curr_frame % SKIP_FRAMES != 0: continue 

                h, w = frame.shape[:2]
                new_h = int(h * (480 / w))
                frame_small = cv2.resize(frame, (480, new_h))
                
                annotated_frame, preds = utils.run_ai_workflow(frame_small)
                
                frame_counts = Counter([p['class'] for p in preds])
                for k, v in frame_counts.items():
                    if v > video_defects[k]:
                        video_defects[k] = v
                
                stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                live_findings.json(dict(video_defects))

            vf.release()
            prog_bar.progress(1.0)
            
            # --- AUTO SAVE LOGIC ---
            final_score, deduction, status = calculate_score(video_defects)
            
            # Simpan Otomatis
            success = db.create_laporan(
                gedung=lokasi_gedung, 
                ruangan=lokasi_ruang, 
                jenis_kerusakan=str(dict(video_defects)), 
                confidence_score=final_score, 
                status=status, 
                deskripsi=f"Auto-Audit Video: {uploaded_video.name}"
            )
            
            if success:
                st.balloons()
                st.success(f"✅ Analisis Selesai! Data Ruangan {lokasi_ruang} telah tersimpan otomatis.")
                
                # Tampilkan Ringkasan Akhir
                c_res1, c_res2, c_res3 = st.columns(3)
                c_res1.metric("Temuan", sum(video_defects.values()))
                c_res2.metric("Skor", f"{final_score}%", f"-{deduction}")
                c_res3.metric("Status", status)
                
                st.info("Silakan cek menu 'Laporan' untuk melihat detail.")
            else:
                st.error("Gagal menyimpan ke database. Cek log terminal.")