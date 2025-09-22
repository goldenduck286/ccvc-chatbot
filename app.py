# app.py - Giao diện web với Streamlit
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
    return model, questions, answers, question_embeddings

model, questions, answers, question_embeddings = load_model_and_data()

# ==================== HÀM TÌM CÂU TRẢ LỜI ====================
def find_best_answer(user_question, top_k=1):
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    cos_scores = util.cos_sim(user_embedding, question_embeddings)[0]
    top_results = np.argpartition(-cos_scores, range(top_k))[:top_k]
    
    results = []
    for idx in top_results:
        score = cos_scores[idx].item()
        results.append({
            'question': questions[idx],
            'answer': answers[idx],
            'similarity': score
        })
    return results

# ==================== GIAO DIỆN WEB ====================
st.set_page_config(page_title="🤖 Chatbot Hỏi Đáp", page_icon="💬")
st.title("💬 Chatbot Tìm Kiếm Câu Trả Lời")

st.markdown("""
Nhập câu hỏi của bạn vào ô bên dưới, bot sẽ tìm và trả lời dựa trên dữ liệu đã được cung cấp.
""")

# Ô nhập liệu trên web — KHÔNG DÙNG TERMINAL
user_input = st.text_input("📝 Bạn hỏi:", placeholder="Ví dụ: Hướng dẫn cấp tài khoản")

# Nút gửi (tùy chọn, để tăng trải nghiệm)
if st.button("Gửi") or user_input:  # Có thể nhấn Enter hoặc nhấn nút
    if user_input.strip() == "":
        st.warning("⚠️ Vui lòng nhập câu hỏi!")
    else:
        with st.spinner("Đang tìm câu trả lời phù hợp..."):
            results = find_best_answer(user_input, top_k=1)
            best = results[0]

            if best['similarity'] > 0.5:
                st.success(f"✅ **Câu trả lời:** {best['answer']}")
                # with st.expander("🔍 Xem chi tiết"):
                #     st.write(f"- Câu hỏi mẫu gần nhất: *{best['question']}*")
                #     st.write(f"- Độ tương đồng: `{best['similarity']:.3f}`")
            else:
                st.error("❌ Xin lỗi, tôi chưa tìm được câu trả lời phù hợp.")

# Footer nhỏ
st.markdown("---")
st.caption("© 2025 Chatbot Hỏi Đáp - Powered by ducnv.hth TT CNTT VNPT Hà Tĩnh")