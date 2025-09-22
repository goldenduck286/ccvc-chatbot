# app.py - Phiên bản HOÀN CHỈNH: Tự động hiển thị PDF nếu là link Google Drive — full width, không cần nhấn xem trước
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import re

# ==================== LOAD MODEL & DATA ====================
@st.cache_resource
def load_model_and_data():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df = pd.read_excel("cau_hoi_tra_loi.xlsx")
    questions = df['Câu hỏi mẫu'].tolist()
    answers = df['Câu trả lời'].tolist()
    question_embeddings = model.encode(questions, convert_to_tensor=True)
    return model, questions, answers, question_embeddings, df

model, all_questions, all_answers, question_embeddings, df = load_model_and_data()

# ==================== HÀM TÌM CÂU TRẢ LỜI ====================
def find_best_answer(user_question, top_k=1):
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    cos_scores = util.cos_sim(user_embedding, question_embeddings)[0]
    top_results = np.argpartition(-cos_scores, range(top_k))[:top_k]
    
    results = []
    for idx in top_results:
        score = cos_scores[idx].item()
        results.append({
            'question': all_questions[idx],
            'answer': all_answers[idx],
            'similarity': score
        })
    return results

# ==================== HÀM XỬ LÝ LINK GOOGLE DRIVE ====================
def is_google_drive_pdf_link(text):
    """Kiểm tra xem chuỗi có phải là link Google Drive đến file PDF không"""
    pattern = r"https://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)/"
    match = re.search(pattern, text)
    return match

def get_direct_download_link(drive_url):
    """Chuyển link Google Drive sang link tải trực tiếp"""
    match = is_google_drive_pdf_link(drive_url)
    if match:
        file_id = match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return drive_url

def get_embed_pdf_link(drive_url):
    """Chuyển link Google Drive sang link xem trước PDF trong iframe"""
    match = is_google_drive_pdf_link(drive_url)
    if match:
        file_id = match.group(1)
        return f"https://drive.google.com/file/d/{file_id}/preview"
    return None

# ==================== GIAO DIỆN WEB ====================
st.set_page_config(page_title="Chatbot Hỏi Đáp", page_icon="💬", layout="wide")
st.title("💬 Chatbot Tìm Kiếm Câu Trả Lời")

st.markdown("""
Vui lòng **chọn một câu hỏi** từ danh sách bên dưới, sau đó nhấn **Gửi** để nhận câu trả lời.
""")

# ➕ SELECTBOX DUY NHẤT — KHÔNG Ô NHẬP LIỆU
chosen_question = st.selectbox(
    "📌 Chọn câu hỏi:",
    options=[""] + all_questions,
    format_func=lambda x: "— Vui lòng chọn —" if x == "" else x,
    index=0,
    key="final_selector"
)

# Nút gửi
if st.button("🔍 Gửi"):
    if not chosen_question.strip():
        st.warning("⚠️ Vui lòng chọn câu hỏi trước khi gửi!")
    else:
        with st.spinner("Đang tìm câu trả lời phù hợp..."):
            results = find_best_answer(chosen_question, top_k=1)
            best = results[0]

            if best['similarity'] > 0.5:
                answer = best['answer']
                st.success("✅ **Câu trả lời:**")

                # Kiểm tra nếu là link Google Drive PDF
                drive_match = is_google_drive_pdf_link(answer)
                if drive_match:
                    file_id = drive_match.group(1)
                    download_link = get_direct_download_link(answer)
                    preview_link = get_embed_pdf_link(answer)

                    # Hiển thị nút tải
                    st.markdown(f"📄 [📥 Tải file PDF]({download_link})")

                    # TỰ ĐỘNG HIỂN THỊ FILE PDF — FULL WIDTH
                    if preview_link:
                        st.components.v1.iframe(
                            src=preview_link,
                            width=None,  # = 100%
                            height=700,
                            scrolling=True
                        )
                else:
                    # Hiển thị text bình thường
                    st.write(answer)
            else:
                st.error("❌ Xin lỗi, tôi chưa tìm được câu trả lời phù hợp.")

# Footer nhỏ
st.markdown("---")
st.caption("© 2025 Chatbot Hỏi Đáp - Powered by ducnv.hth TT CNTT VNPT Hà Tĩnh")