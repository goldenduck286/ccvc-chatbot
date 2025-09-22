import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load mô hình mã hóa câu
model = SentenceTransformer('all-MiniLM-L6-v2')

# Đọc dữ liệu từ Excel
df = pd.read_excel("cau_hoi_tra_loi.xlsx")
questions = df['Câu hỏi mẫu'].tolist()
answers = df['Câu trả lời'].tolist()

# Mã hóa tất cả câu hỏi mẫu thành vector
question_embeddings = model.encode(questions, convert_to_tensor=True)

def find_best_answer(user_question, top_k=1):
    # Mã hóa câu hỏi người dùng
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    
    # Tính độ tương đồng cosine
    cos_scores = util.cos_sim(user_embedding, question_embeddings)[0]
    
    # Lấy top K kết quả
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

# Vòng lặp chat bot đơn giản
print("Bot: Xin chào! Bạn có thể hỏi tôi bất kỳ câu hỏi nào.")
while True:
    user_input = input("Bạn: ")
    if user_input.lower() in ['quit', 'exit', 'thoát']:
        print("Bot: Tạm biệt!")
        break
    
    best_answers = find_best_answer(user_input, top_k=1)
    best = best_answers[0]
    
    if best['similarity'] > 0.5:  # Ngưỡng tương đồng, có thể điều chỉnh
        print(f"Bot: {best['answer']} (Độ tương đồng: {best['similarity']:.2f})")
    else:
        print("Bot: Xin lỗi, tôi chưa hiểu câu hỏi của bạn.")