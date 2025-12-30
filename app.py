import streamlit as st
import cv2
import numpy as np

# ================== ĐÁP ÁN ==================
# 0=A, 1=B, 2=C, 3=D
ANSWER_KEY = [
    0,1,2,3,0,1,2,3,0,1,
    2,3,1,0,2,3,1,0,2,3,
    0,1,2,3,0,1,2,3,0,1,
    2,3,1,0,2,3,1,0,2,3
]

# ================== HÀM OMR ==================
def detect_answers(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    answers = []

    # ===== TOẠ ĐỘ MẪU – CHỈNH THEO PHIẾU =====
    start_x = 120    # cột A
    start_y = 350    # câu 1
    row_gap = 35     # khoảng cách câu
    col_gap = 40     # A-B-C-D
    bubble = 22      # kích thước ô tròn

    for q in range(40):
        y = start_y + q * row_gap
        values = []
        for c in range(4):
            x = start_x + c * col_gap
            roi = thresh[y:y+bubble, x:x+bubble]
            values.append(cv2.countNonZero(roi))
        answers.append(int(np.argmax(values)))

    return answers

# ================== STREAMLIT UI ==================
st.set_page_config(
    page_title="Chấm trắc nghiệm",
    layout="centered"
)

st.title("📄 Ứng dụng chấm trắc nghiệm ABCD (40 câu)")
st.write("Upload ảnh phiếu → bấm **Chấm bài**")

uploaded = st.file_uploader(
    "📷 Tải ảnh phiếu trả lời",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    file_bytes = np.asarray(
        bytearray(uploaded.read()),
        dtype=np.uint8
    )
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(img, caption="Ảnh phiếu trả lời", channels="BGR")

    if st.button("🧮 Chấm bài"):
        student_answers = detect_answers(img)

        correct = sum(
            a == b for a, b in zip(student_answers, ANSWER_KEY)
        )
        score = round(correct / 40 * 10, 2)

        st.success(f"✅ Số câu đúng: {correct} / 40")
        st.success(f"🎯 Điểm: {score}")

        st.write("### 📋 Chi tiết đáp án:")
        for i, ans in enumerate(student_answers):
            st.write(
                f"Câu {i+1}: "
                f"{['A','B','C','D'][ans]} "
                f"{'✔' if ans == ANSWER_KEY[i] else '✘'}"
            )

st.info(
    "💡 Lưu ý: nếu chấm lệch, chỉ cần chỉnh "
    "start_x, start_y, row_gap, col_gap trong code."
)
