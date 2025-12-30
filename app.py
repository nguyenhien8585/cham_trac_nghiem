import streamlit as st
import numpy as np
import cv2

# ================== CẤU HÌNH ==================
st.set_page_config(page_title="Chấm trắc nghiệm", layout="centered")
st.title("📸 Chấm trắc nghiệm bằng Camera (OMR)")

st.write("👉 Đưa phiếu vào camera → chụp → chấm ngay")

# ================== ĐÁP ÁN MẪU ==================
# PHẦN I: 40 câu ABCD
ANSWER_ABCD = [
    0,1,2,3,0,1,2,3,0,1,
    2,3,1,0,2,3,1,0,2,3,
    0,1,2,3,0,1,2,3,0,1,
    2,3,1,0,2,3,1,0,2,3
]

# PHẦN II: Đúng(True) / Sai(False) – 8 câu, mỗi câu 4 ý
ANSWER_TRUEFALSE = [
    [1,0,1,1],
    [1,1,0,0],
    [0,1,1,0],
    [1,0,0,1],
    [1,1,1,0],
    [0,0,1,1],
    [1,0,1,0],
    [0,1,0,1]
]

# ================== TIỀN XỬ LÝ ==================
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    return thresh

# ================== OMR ABCD ==================
def detect_abcd(thresh):
    answers = []

    start_x = 120
    start_y = 350
    row_gap = 35
    col_gap = 40
    bubble = 22

    for q in range(40):
        y = start_y + q * row_gap
        filled = []
        for c in range(4):
            x = start_x + c * col_gap
            roi = thresh[y:y+bubble, x:x+bubble]
            filled.append(cv2.countNonZero(roi))
        answers.append(int(np.argmax(filled)))

    return answers

# ================== OMR ĐÚNG / SAI ==================
def detect_truefalse(thresh):
    results = []

    start_x = 350
    start_y = 350
    row_gap = 45
    col_gap = 35
    bubble = 20

    for q in range(8):
        row = []
        y = start_y + q * row_gap
        for i in range(4):
            x_true = start_x + i * col_gap
            x_false = x_true + 18

            roi_true = thresh[y:y+bubble, x_true:x_true+bubble]
            roi_false = thresh[y:y+bubble, x_false:x_false+bubble]

            row.append(1 if cv2.countNonZero(roi_true) >
                           cv2.countNonZero(roi_false) else 0)
        results.append(row)

    return results

# ================== CAMERA INPUT ==================
image_file = st.camera_input("📷 Chụp phiếu trả lời")

if image_file:
    bytes_data = image_file.getvalue()
    img = cv2.imdecode(
        np.frombuffer(bytes_data, np.uint8),
        cv2.IMREAD_COLOR
    )

    st.image(img, caption="Ảnh vừa chụp", channels="BGR")

    thresh = preprocess(img)

    # ---- CHẤM PHẦN I ----
    abcd_student = detect_abcd(thresh)
    abcd_correct = sum(
        a == b for a, b in zip(abcd_student, ANSWER_ABCD)
    )

    # ---- CHẤM PHẦN II ----
    tf_student = detect_truefalse(thresh)
    tf_correct = 0
    for i in range(8):
        for j in range(4):
            if tf_student[i][j] == ANSWER_TRUEFALSE[i][j]:
                tf_correct += 1

    # ---- TÍNH ĐIỂM ----
    total_correct = abcd_correct + tf_correct
    total_questions = 40 + 32
    score = round(total_correct / total_questions * 10, 2)

    st.success(f"✅ Tổng số ý đúng: {total_correct}")
    st.success(f"🎯 Điểm: {score}")

    st.write("### 📋 Chi tiết Phần I (ABCD)")
    for i, a in enumerate(abcd_student):
        st.write(
            f"Câu {i+1}: {['A','B','C','D'][a]} "
            f"{'✔' if a == ANSWER_ABCD[i] else '✘'}"
        )

    st.write("### 📋 Chi tiết Phần II (Đúng / Sai)")
    for i in range(8):
        st.write(f"Câu {i+1}: {tf_student[i]}")

st.info(
    "📌 Nếu chấm lệch: chỉnh các biến start_x, start_y, row_gap, col_gap trong code để khớp đúng phiếu."
)
