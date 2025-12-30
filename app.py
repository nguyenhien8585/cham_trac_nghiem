import streamlit as st
import numpy as np
import cv2

# ================== CẤU HÌNH APP ==================
st.set_page_config(
    page_title="Chấm trắc nghiệm",
    layout="centered"
)

st.title("📄 Ứng dụng chấm trắc nghiệm ABCD (40 câu)")
st.write("Tải ảnh phiếu trả lời → Bấm **Chấm bài**")

# ================== ĐÁP ÁN ==================
# 0 = A, 1 = B, 2 = C, 3 = D
ANSWER_KEY = [
    0,1,2,3,0,1,2,3,0,1,
    2,3,1,0,2,3,1,0,2,3,
    0,1,2,3,0,1,2,3,0,1,
    2,3,1,0,2,3,1,0,2,3
]

# ================== HÀM NHẬN DẠNG OMR ==================
def detect_answers(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    answers = []

    # ===== TOẠ ĐỘ MẪU (CẦN CHỈNH CHO ĐÚNG PHIẾU) =====
    start_x = 120     # vị trí cột A
    start_y = 350     # vị trí câu 1
    row_gap = 35      # khoảng cách giữa các câu
    col_gap = 40      # khoảng cách A-B-C-D
    bubble_size = 22  # kích thước ô tô

    for q in range(40):
        y = start_y + q * row_gap
        values = []

        for c in range(4):
            x = start_x + c * col_gap
            roi = thresh[y:y+bubble_size, x:x+bubble_size]

            if roi.size == 0:
                values.append(0)
            else:
                values.append(cv2.countNonZero(roi))

        answers.append(int(np.argmax(values)))

    return answers

# ================== GIAO DIỆN ==================
uploaded_file = st.file_uploader(
    "📷 Tải ảnh phiếu trả lời",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image, caption="Ảnh phiếu trả lời", channels="BGR")

    if st.button("🧮 Chấm bài"):
        student_answers = detect_answers(image)

        correct = sum(
            s == k for s, k in zip(student_answers, ANSWER_KEY)
        )
        score = round(correct / 40 * 10, 2)

        st.success(f"✅ Số câu đúng: {correct} / 40")
        st.success(f"🎯 Điểm: {score}")

        st.write("### 📋 Chi tiết bài làm")
        for i, ans in enumerate(student_answers):
            st.write(
                f"Câu {i+1}: "
                f"{['A','B','C','D'][ans]} "
                f"{'✔' if ans == ANSWER_KEY[i] else '✘'}"
            )

st.info(
    "ℹ️ Nếu chấm chưa đúng, hãy chỉnh các giá trị: "
    "start_x, start_y, row_gap, col_gap, bubble_size trong code."
)
