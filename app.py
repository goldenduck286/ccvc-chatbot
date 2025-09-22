# app.py - Phiên bản CUỐI CÙNG: Chỉ dùng selectbox, không có ô nhập liệu
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np

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

# ==================== GIAO DIỆN WEB ====================
st.set_page_config(page_title="Chatbot Hỏi Đáp", page_icon="💬")
st.title("💬 Chatbot Tìm Kiếm Câu Trả Lời")

st.markdown("""
Vui lòng **chọn một câu hỏi** từ danh sách bên dưới, sau đó nhấn **Gửi** để nhận câu trả lời.
""")

# ➕ CHỈ DÙNG SELECTBOX — KHÔNG CÓ Ô NHẬP LIỆU
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
                st.success(f"✅ **Câu trả lời:** {best['answer']}")
            else:
                st.error("❌ Xin lỗi, tôi chưa tìm được câu trả lời phù hợp.")

# Footer nhỏ
st.markdown("---")
st.caption("© 2025 Chatbot Hỏi Đáp - Powered by ducnv.hth@vnpt.vn TT CNTT VNPT Hà Tĩnh")