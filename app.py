"""
ỨNG DỤNG CHẤM TRẮC NGHIỆM TỰ ĐỘNG BẰNG CAMERA
Sử dụng Streamlit + OpenCV + OMR

Tác giả: STK
Phiên bản: 1.0
"""

import streamlit as st
import cv2
import numpy as np
from typing import List, Tuple, Optional

# ============================================================================
# CẤU HÌNH THAM SỐ - Giáo viên có thể chỉnh tại đây
# ============================================================================

# Số câu hỏi
TOTAL_QUESTIONS = 40

# Điểm tối đa
MAX_SCORE = 10.0

# Tọa độ bắt đầu vùng đáp án (pixel)
# Chỉnh theo vị trí thực tế trên phiếu của bạn
START_X = 100  # Tọa độ X bắt đầu cột A
START_Y = 150  # Tọa độ Y bắt đầu câu 1

# Kích thước mỗi ô tròn (pixel)
BUBBLE_WIDTH = 40   # Chiều rộng ô
BUBBLE_HEIGHT = 35  # Chiều cao ô

# Khoảng cách giữa các ô
COL_GAP = 45   # Khoảng cách giữa A-B-C-D (ngang)
ROW_GAP = 38   # Khoảng cách giữa các câu (dọc)

# Số câu trên mỗi cột (phiếu 40 câu thường chia 2 cột)
QUESTIONS_PER_COLUMN = 20

# Khoảng cách giữa 2 cột
COLUMN_SPACING = 400

# Ngưỡng để xác định ô được tô (% pixel đen)
FILL_THRESHOLD = 0.15  # 15% pixel đen = đã tô

# ============================================================================
# HÀM XỬ LÝ ẢNH
# ============================================================================

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Tiền xử lý ảnh: chuyển grayscale, làm mờ, threshold
    
    Args:
        image: Ảnh gốc (BGR hoặc RGB)
    
    Returns:
        Ảnh nhị phân (đen/trắng)
    """
    # Chuyển sang grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Làm mờ để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold Otsu + Binary Invert (đen = tô, trắng = không tô)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    return thresh


def get_bubble_position(question_num: int, option_index: int) -> Tuple[int, int]:
    """
    Tính tọa độ (x, y) của ô tròn dựa vào số câu và đáp án
    
    Args:
        question_num: Số thứ tự câu hỏi (0-39)
        option_index: Index đáp án (0=A, 1=B, 2=C, 3=D)
    
    Returns:
        Tuple (x, y) tọa độ góc trên-trái của ô
    """
    # Xác định cột (0 hoặc 1)
    column = question_num // QUESTIONS_PER_COLUMN
    row_in_column = question_num % QUESTIONS_PER_COLUMN
    
    # Tính tọa độ
    x = START_X + (option_index * COL_GAP) + (column * COLUMN_SPACING)
    y = START_Y + (row_in_column * ROW_GAP)
    
    return x, y


def extract_bubble(image: np.ndarray, x: int, y: int) -> np.ndarray:
    """
    Cắt vùng ô tròn từ ảnh
    
    Args:
        image: Ảnh đã threshold
        x, y: Tọa độ góc trên-trái
    
    Returns:
        Vùng ảnh của ô tròn
    """
    return image[y:y+BUBBLE_HEIGHT, x:x+BUBBLE_WIDTH]


def is_bubble_filled(bubble: np.ndarray) -> float:
    """
    Tính tỷ lệ pixel đen trong ô để xác định có được tô hay không
    
    Args:
        bubble: Vùng ảnh của ô tròn
    
    Returns:
        Tỷ lệ pixel đen (0.0 - 1.0)
    """
    total_pixels = bubble.shape[0] * bubble.shape[1]
    filled_pixels = cv2.countNonZero(bubble)
    return filled_pixels / total_pixels if total_pixels > 0 else 0.0


def detect_answer(image: np.ndarray, question_num: int) -> Optional[str]:
    """
    Phát hiện đáp án được chọn cho một câu hỏi
    
    Args:
        image: Ảnh đã threshold
        question_num: Số thứ tự câu (0-39)
    
    Returns:
        Đáp án ('A', 'B', 'C', 'D') hoặc None nếu không tô
    """
    options = ['A', 'B', 'C', 'D']
    fill_ratios = []
    
    # Kiểm tra từng ô A, B, C, D
    for i in range(4):
        x, y = get_bubble_position(question_num, i)
        bubble = extract_bubble(image, x, y)
        fill_ratio = is_bubble_filled(bubble)
        fill_ratios.append(fill_ratio)
    
    # Tìm ô có tỷ lệ đen cao nhất
    max_fill = max(fill_ratios)
    
    # Nếu tỷ lệ đen vượt ngưỡng => đã tô
    if max_fill >= FILL_THRESHOLD:
        max_index = fill_ratios.index(max_fill)
        return options[max_index]
    
    return None  # Không có ô nào được tô


def grade_answers(student_answers: List[Optional[str]], 
                   correct_answers: List[str]) -> Tuple[int, float, List[dict]]:
    """
    Chấm điểm bài làm
    
    Args:
        student_answers: Đáp án học sinh (có thể None)
        correct_answers: Đáp án đúng
    
    Returns:
        Tuple (số câu đúng, điểm, chi tiết từng câu)
    """
    correct_count = 0
    details = []
    
    for i in range(TOTAL_QUESTIONS):
        student_ans = student_answers[i] if i < len(student_answers) else None
        correct_ans = correct_answers[i] if i < len(correct_answers) else None
        
        is_correct = (student_ans == correct_ans) and (student_ans is not None)
        if is_correct:
            correct_count += 1
        
        details.append({
            'question': i + 1,
            'student_answer': student_ans or '(Chưa chọn)',
            'correct_answer': correct_ans or '?',
            'is_correct': is_correct
        })
    
    # Tính điểm theo thang 10
    score = (correct_count / TOTAL_QUESTIONS) * MAX_SCORE
    
    return correct_count, score, details


# ============================================================================
# HÀM XỬ LÝ ĐẦU VÀO
# ============================================================================

def parse_answer_key(text: str) -> Optional[List[str]]:
    """
    Phân tích chuỗi đáp án từ giáo viên
    
    Args:
        text: Chuỗi đáp án (VD: "ABCDABCD...")
    
    Returns:
        List các đáp án hoặc None nếu lỗi
    """
    # Xóa khoảng trắng, xuống dòng
    text = text.replace(' ', '').replace('\n', '').replace('\r', '').upper()
    
    # Lọc chỉ giữ A, B, C, D
    answers = [c for c in text if c in 'ABCD']
    
    # Kiểm tra đủ 40 đáp án
    if len(answers) != TOTAL_QUESTIONS:
        return None
    
    return answers


# ============================================================================
# GIAO DIỆN STREAMLIT
# ============================================================================

def main():
    st.set_page_config(
        page_title="Chấm Trắc Nghiệm Tự Động",
        page_icon="📝",
        layout="wide"
    )
    
    # Header
    st.title("📝 CHẤM TRẮC NGHIỆM TỰ ĐỘNG")
    st.markdown("**Sử dụng camera để chụp phiếu trả lời và chấm điểm ngay lập tức**")
    st.divider()
    
    # Sidebar - Hướng dẫn
    with st.sidebar:
        st.header("📖 Hướng Dẫn Sử Dụng")
        st.markdown("""
        ### Bước 1: Nhập đáp án đúng
        - Nhập 40 đáp án (A, B, C hoặc D)
        - Có thể viết liền hoặc xuống dòng
        - Ví dụ: `ABCDABCD...` hoặc từng câu 1 dòng
        
        ### Bước 2: Chụp phiếu
        - Đặt phiếu trả lời phẳng, rõ nét
        - Đủ ánh sáng, không bóng mờ
        - Chụp vuông góc
        
        ### Bước 3: Xem kết quả
        - Số câu đúng
        - Điểm số
        - Chi tiết từng câu
        """)
        
        st.divider()
        st.info(f"**Tham số hiện tại:**\n- Số câu: {TOTAL_QUESTIONS}\n- Điểm tối đa: {MAX_SCORE}")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("1️⃣ Nhập Đáp Án Đúng")
        
        answer_input = st.text_area(
            "Nhập 40 đáp án (A/B/C/D):",
            height=150,
            placeholder="Ví dụ:\nABCDABCDABCD...\nhoặc:\nA\nB\nC\nD\n...",
            help="Nhập đúng 40 ký tự A, B, C hoặc D. Có thể có khoảng trắng/xuống dòng."
        )
        
        # Parse đáp án
        correct_answers = None
        if answer_input.strip():
            correct_answers = parse_answer_key(answer_input)
            
            if correct_answers:
                st.success(f"✅ Đã nhập đủ {len(correct_answers)} đáp án!")
                with st.expander("Xem đáp án đã nhập"):
                    # Hiển thị dạng bảng
                    for i in range(0, TOTAL_QUESTIONS, 10):
                        row = [f"**{j+1}.** {correct_answers[j]}" 
                               for j in range(i, min(i+10, TOTAL_QUESTIONS))]
                        st.markdown(" | ".join(row))
            else:
                st.error(f"❌ Lỗi: Cần đúng {TOTAL_QUESTIONS} đáp án, bạn đã nhập {len(answer_input.replace(' ', '').replace('\\n', ''))} ký tự hợp lệ.")
    
    with col2:
        st.header("2️⃣ Chụp Phiếu Trả Lời")
        
        if not correct_answers:
            st.warning("⚠️ Vui lòng nhập đáp án đúng trước khi chụp ảnh!")
        else:
            # Camera input
            camera_photo = st.camera_input("Chụp phiếu trả lời:")
            
            if camera_photo is not None:
                # Đọc ảnh từ camera
                file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                # Hiển thị ảnh gốc
                st.image(camera_photo, caption="Ảnh đã chụp", use_container_width=True)
                
                # Xử lý và chấm điểm
                with st.spinner("🔍 Đang phân tích phiếu..."):
                    # Tiền xử lý ảnh
                    processed = preprocess_image(image)
                    
                    # Phát hiện đáp án từng câu
                    student_answers = []
                    for q in range(TOTAL_QUESTIONS):
                        answer = detect_answer(processed, q)
                        student_answers.append(answer)
                    
                    # Chấm điểm
                    correct_count, score, details = grade_answers(student_answers, correct_answers)
                
                # Hiển thị kết quả
                st.divider()
                st.header("3️⃣ Kết Quả Chấm Điểm")
                
                # Thống kê tổng quan
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("📊 Số câu đúng", f"{correct_count}/{TOTAL_QUESTIONS}")
                
                with metric_col2:
                    st.metric("⭐ Điểm số", f"{score:.2f}/{MAX_SCORE}")
                
                with metric_col3:
                    percent = (correct_count / TOTAL_QUESTIONS) * 100
                    st.metric("📈 Tỷ lệ đúng", f"{percent:.1f}%")
                
                # Chi tiết từng câu
                st.divider()
                st.subheader("📋 Chi Tiết Từng Câu")
                
                # Hiển thị dạng bảng
                for i in range(0, TOTAL_QUESTIONS, 10):
                    cols = st.columns(10)
                    for j in range(10):
                        idx = i + j
                        if idx < TOTAL_QUESTIONS:
                            detail = details[idx]
                            with cols[j]:
                                if detail['is_correct']:
                                    st.markdown(f"**{idx+1}** ✅")
                                    st.success(detail['student_answer'])
                                else:
                                    st.markdown(f"**{idx+1}** ❌")
                                    st.error(f"{detail['student_answer']}\n({detail['correct_answer']})")
                
                # Danh sách câu sai
                wrong_questions = [d for d in details if not d['is_correct']]
                if wrong_questions:
                    st.divider()
                    st.subheader(f"❌ Các Câu Sai ({len(wrong_questions)} câu)")
                    
                    wrong_text = ", ".join([str(d['question']) for d in wrong_questions])
                    st.error(f"**Câu sai:** {wrong_text}")
                    
                    with st.expander("Xem chi tiết câu sai"):
                        for d in wrong_questions:
                            st.markdown(
                                f"**Câu {d['question']}:** "
                                f"Chọn {d['student_answer']} → "
                                f"Đúng là {d['correct_answer']}"
                            )
                else:
                    st.balloons()
                    st.success("🎉 **HOÀN HẢO!** Tất cả câu đều đúng!")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <small>
        💡 <b>Lưu ý:</b> Nếu kết quả không chính xác, hãy điều chỉnh các tham số 
        <code>START_X, START_Y, COL_GAP, ROW_GAP, BUBBLE_WIDTH, BUBBLE_HEIGHT</code> 
        trong code để khớp với phiếu của bạn.
        </small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
