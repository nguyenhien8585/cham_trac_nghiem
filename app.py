"""
ỨNG DỤNG CHẤM TRẮC NGHIỆM TỰ ĐỘNG - PHIẾU CHUẨN BỘ GD&ĐT
Hỗ trợ đầy đủ 3 phần: ABCD + Đúng/Sai + Điền số
Sử dụng Streamlit + OpenCV + OMR

Tác giả: STK
Phiên bản: 2.0 (Nâng cao)
"""

import streamlit as st
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import json

# ============================================================================
# CẤU HÌNH THAM SỐ - Giáo viên có thể chỉnh tại đây
# ============================================================================

# === TỌA ĐỘ VÀ KÍCH THƯỚC - Chỉnh theo phiếu của bạn ===

# Phần I: Trắc nghiệm ABCD
PART1_LAYOUT = {
    'start_x': 50,              # Tọa độ X bắt đầu cột 1
    'start_y': 180,             # Tọa độ Y bắt đầu câu 1
    'bubble_width': 28,         # Chiều rộng ô tròn
    'bubble_height': 28,        # Chiều cao ô tròn
    'col_gap': 35,              # Khoảng cách A→B→C→D
    'row_gap': 35,              # Khoảng cách câu 1→2→3
    'column_spacing': 300,      # Khoảng cách cột 1→cột 2
}

# Phần II: Đúng/Sai
PART2_LAYOUT = {
    'items_per_question': 4,    # Mỗi câu có 4 ý (a,b,c,d) - CỐ ĐỊNH
    'start_x': 600,             # Tọa độ X bắt đầu Câu 1
    'start_y': 180,             # Tọa độ Y bắt đầu
    'bubble_width': 20,         # Kích thước ô Đúng/Sai
    'bubble_height': 20,
    'col_gap': 35,              # Khoảng cách Đúng→Sai
    'row_gap': 25,              # Khoảng cách a→b→c→d
    'question_gap': 140,        # Khoảng cách Câu 1→Câu 2
}

# Phần III: Điền số
PART3_LAYOUT = {
    'digits_per_question': 6,   # Mỗi câu tối đa 6 ký tự - CỐ ĐỊNH
    'digit_options': 11,        # 0-9 + dấu phẩy = 11 option - CỐ ĐỊNH
    'start_x': 900,             # Tọa độ X bắt đầu Câu 1
    'start_y': 180,             # Tọa độ Y bắt đầu
    'bubble_width': 18,         # Kích thước ô số
    'bubble_height': 18,
    'col_gap': 22,              # Khoảng cách giữa các vị trí
    'row_gap': 20,              # Khoảng cách 0→1→2→...
    'question_gap': 160,        # Khoảng cách Câu 1→Câu 2
}

# Ngưỡng nhận diện ô được tô
FILL_THRESHOLD = 0.20  # 20% pixel đen = đã tô

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


def extract_bubble(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """
    Cắt vùng ô tròn từ ảnh
    
    Args:
        image: Ảnh đã threshold
        x, y: Tọa độ góc trên-trái
        width, height: Kích thước ô
    
    Returns:
        Vùng ảnh của ô tròn
    """
    return image[y:y+height, x:x+width]


def is_bubble_filled(bubble: np.ndarray) -> float:
    """
    Tính tỷ lệ pixel đen trong ô để xác định có được tô hay không
    
    Args:
        bubble: Vùng ảnh của ô tròn
    
    Returns:
        Tỷ lệ pixel đen (0.0 - 1.0)
    """
    if bubble.size == 0:
        return 0.0
    total_pixels = bubble.shape[0] * bubble.shape[1]
    filled_pixels = cv2.countNonZero(bubble)
    return filled_pixels / total_pixels if total_pixels > 0 else 0.0


# ============================================================================
# PHẦN I: Trắc nghiệm ABCD
# ============================================================================

def detect_part1_answer(image: np.ndarray, question_num: int, config: dict) -> Optional[str]:
    """
    Phát hiện đáp án ABCD cho một câu hỏi ở Phần I
    
    Args:
        image: Ảnh đã threshold
        question_num: Số thứ tự câu (0-based)
        config: Dict chứa thông tin cấu hình
    
    Returns:
        Đáp án ('A', 'B', 'C', 'D') hoặc None
    """
    layout = PART1_LAYOUT
    total_questions = config['total_questions']
    questions_per_column = config['questions_per_column']
    
    options = ['A', 'B', 'C', 'D']
    fill_ratios = []
    
    # Xác định cột (0 hoặc 1)
    column = question_num // questions_per_column
    row_in_column = question_num % questions_per_column
    
    # Kiểm tra từng ô A, B, C, D
    for i in range(4):
        x = layout['start_x'] + (i * layout['col_gap']) + (column * layout['column_spacing'])
        y = layout['start_y'] + (row_in_column * layout['row_gap'])
        
        bubble = extract_bubble(image, x, y, layout['bubble_width'], layout['bubble_height'])
        fill_ratio = is_bubble_filled(bubble)
        fill_ratios.append(fill_ratio)
    
    # Tìm ô có tỷ lệ đen cao nhất
    max_fill = max(fill_ratios)
    
    if max_fill >= FILL_THRESHOLD:
        max_index = fill_ratios.index(max_fill)
        return options[max_index]
    
    return None


def grade_part1(student_answers: List[Optional[str]], 
                correct_answers: List[str],
                config: dict) -> Tuple[int, List[dict]]:
    """
    Chấm điểm Phần I
    
    Args:
        student_answers: Đáp án học sinh
        correct_answers: Đáp án đúng
        config: Dict chứa cấu hình (total_questions, score)
    
    Returns:
        Tuple (số câu đúng, chi tiết từng câu)
    """
    correct_count = 0
    details = []
    total_questions = config['total_questions']
    
    for i in range(total_questions):
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
    
    return correct_count, details


# ============================================================================
# PHẦN II: Đúng/Sai
# ============================================================================

def detect_part2_answer(image: np.ndarray, question_num: int, item_index: int, config: dict) -> Optional[str]:
    """
    Phát hiện đáp án Đúng/Sai cho một ý của câu hỏi ở Phần II
    
    Args:
        image: Ảnh đã threshold
        question_num: Số câu (0-based)
        item_index: Số ý (0=a, 1=b, 2=c, 3=d)
        config: Dict chứa cấu hình
    
    Returns:
        'Đúng', 'Sai' hoặc None
    """
    layout = PART2_LAYOUT
    options = ['Đúng', 'Sai']
    fill_ratios = []
    
    # Kiểm tra ô Đúng và Sai
    for i in range(2):
        x = layout['start_x'] + (i * layout['col_gap'])
        y = layout['start_y'] + (question_num * layout['question_gap']) + (item_index * layout['row_gap'])
        
        bubble = extract_bubble(image, x, y, layout['bubble_width'], layout['bubble_height'])
        fill_ratio = is_bubble_filled(bubble)
        fill_ratios.append(fill_ratio)
    
    max_fill = max(fill_ratios)
    
    if max_fill >= FILL_THRESHOLD:
        max_index = fill_ratios.index(max_fill)
        return options[max_index]
    
    return None


def grade_part2(student_answers: List[List[Optional[str]]], 
                correct_answers: List[List[str]],
                config: dict) -> Tuple[int, List[dict]]:
    """
    Chấm điểm Phần II
    
    Args:
        student_answers: Đáp án học sinh [[a,b,c,d], [a,b,c,d], ...]
        correct_answers: Đáp án đúng
        config: Dict chứa cấu hình (total_questions, score)
    
    Returns:
        Tuple (số ý đúng, chi tiết)
    """
    correct_count = 0
    details = []
    items_labels = ['a', 'b', 'c', 'd']
    items_per_question = PART2_LAYOUT['items_per_question']
    total_questions = config['total_questions']
    
    for q in range(total_questions):
        for i in range(items_per_question):
            student_ans = student_answers[q][i] if q < len(student_answers) and i < len(student_answers[q]) else None
            correct_ans = correct_answers[q][i] if q < len(correct_answers) and i < len(correct_answers[q]) else None
            
            is_correct = (student_ans == correct_ans) and (student_ans is not None)
            if is_correct:
                correct_count += 1
            
            details.append({
                'question': f"Câu {q+1}{items_labels[i]}",
                'student_answer': student_ans or '(Chưa chọn)',
                'correct_answer': correct_ans or '?',
                'is_correct': is_correct
            })
    
    return correct_count, details


# ============================================================================
# PHẦN III: Điền số
# ============================================================================

def detect_part3_answer(image: np.ndarray, question_num: int, config: dict) -> Optional[str]:
    """
    Phát hiện số điền vào cho một câu ở Phần III
    
    Args:
        image: Ảnh đã threshold
        question_num: Số câu (0-based)
        config: Dict chứa cấu hình
    
    Returns:
        Chuỗi số (VD: "-1,5", "123") hoặc None
    """
    layout = PART3_LAYOUT
    digit_map = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ',']
    result = []
    
    # Duyệt qua 6 vị trí (từ trái sang phải)
    for pos in range(layout['digits_per_question']):
        fill_ratios = []
        
        # Kiểm tra 11 option (0-9 + dấu phẩy)
        for digit_idx in range(layout['digit_options']):
            x = layout['start_x'] + (pos * layout['col_gap'])
            y = layout['start_y'] + (question_num * layout['question_gap']) + (digit_idx * layout['row_gap'])
            
            bubble = extract_bubble(image, x, y, layout['bubble_width'], layout['bubble_height'])
            fill_ratio = is_bubble_filled(bubble)
            fill_ratios.append(fill_ratio)
        
        max_fill = max(fill_ratios)
        
        if max_fill >= FILL_THRESHOLD:
            max_index = fill_ratios.index(max_fill)
            result.append(digit_map[max_index])
        else:
            # Nếu không có ô nào được tô, dừng lại (bỏ trống các ô sau)
            break
    
    return ''.join(result) if result else None


def grade_part3(student_answers: List[Optional[str]], 
                correct_answers: List[str],
                config: dict) -> Tuple[int, List[dict]]:
    """
    Chấm điểm Phần III
    
    Args:
        student_answers: Đáp án học sinh
        correct_answers: Đáp án đúng
        config: Dict chứa cấu hình (total_questions, score)
    
    Returns:
        Tuple (số câu đúng, chi tiết)
    """
    correct_count = 0
    details = []
    total_questions = config['total_questions']
    
    for i in range(total_questions):
        student_ans = student_answers[i] if i < len(student_answers) else None
        correct_ans = correct_answers[i] if i < len(correct_answers) else None
        
        is_correct = (student_ans == correct_ans) and (student_ans is not None)
        if is_correct:
            correct_count += 1
        
        details.append({
            'question': f"Câu {i+1}",
            'student_answer': student_ans or '(Chưa điền)',
            'correct_answer': correct_ans or '?',
            'is_correct': is_correct
        })
    
    return correct_count, details


# ============================================================================
# HÀM XỬ LÝ ĐẦU VÀO
# ============================================================================

def parse_part1_answers(text: str, total_questions: int) -> Optional[List[str]]:
    """
    Phân tích đáp án Phần I (ABCD)
    
    Args:
        text: Chuỗi đáp án (VD: "ABCDABCD...")
        total_questions: Số câu hỏi
    
    Returns:
        List các đáp án hoặc None nếu lỗi
    """
    text = text.replace(' ', '').replace('\n', '').replace('\r', '').upper()
    answers = [c for c in text if c in 'ABCD']
    
    if len(answers) != total_questions:
        return None
    
    return answers


def parse_part2_answers(text: str, total_questions: int) -> Optional[List[List[str]]]:
    """
    Phân tích đáp án Phần II (Đúng/Sai)
    
    Args:
        text: Chuỗi đáp án, mỗi câu 1 dòng
        Format: "Đ S Đ Đ" hoặc "D S D D" (4 ý cho mỗi câu)
        total_questions: Số câu hỏi
    
    Returns:
        List [[a,b,c,d], [a,b,c,d], ...] hoặc None nếu lỗi
    """
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    
    if len(lines) != total_questions:
        return None
    
    answers = []
    items_per_question = PART2_LAYOUT['items_per_question']  # Luôn là 4 (a,b,c,d)
    
    for line in lines:
        # Thay thế chữ viết tắt
        line = line.upper().replace('Đ', 'Đúng').replace('D', 'Đúng').replace('S', 'Sai')
        
        # Tách các từ
        words = line.split()
        items = []
        for word in words:
            if 'Đ' in word or word.startswith('T'):  # Đúng hoặc True
                items.append('Đúng')
            elif word.startswith('S') or word.startswith('F'):  # Sai hoặc False
                items.append('Sai')
        
        if len(items) != items_per_question:
            return None
        
        answers.append(items)
    
    return answers


def parse_part3_answers(text: str, total_questions: int) -> Optional[List[str]]:
    """
    Phân tích đáp án Phần III (Điền số)
    
    Args:
        text: Chuỗi đáp án, mỗi câu 1 dòng
        Format: "-1,5" hoặc "123"
        total_questions: Số câu hỏi
    
    Returns:
        List các số hoặc None nếu lỗi
    """
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    
    if len(lines) != total_questions:
        return None
    
    return lines


# ============================================================================
# GIAO DIỆN STREAMLIT
# ============================================================================

def main():
    st.set_page_config(
        page_title="Chấm Trắc Nghiệm - Phiếu Chuẩn Bộ GD&ĐT",
        page_icon="📝",
        layout="wide"
    )
    
    # Header
    st.title("📝 CHẤM TRẮC NGHIỆM TỰ ĐỘNG")
    st.markdown("**Phiếu chuẩn Bộ GD&ĐT - Hỗ trợ đầy đủ 3 phần**")
    st.divider()
    
    # Sidebar - Cấu hình
    with st.sidebar:
        st.header("⚙️ Cấu Hình")
        
        # Chọn camera
        st.subheader("📷 Cấu hình camera")
        camera_option = st.radio(
            "Chọn camera:",
            options=["Camera trước (Front)", "Camera sau (Back)"],
            index=0,
            help="Camera sau thường có chất lượng tốt hơn"
        )
        use_back_camera = "sau" in camera_option.lower()
        
        st.divider()
        
        # Chọn phần cần chấm
        st.subheader("📚 Chọn phần cần chấm")
        enable_part1 = st.checkbox("✅ Phần I: Trắc nghiệm ABCD", value=True)
        enable_part2 = st.checkbox("✅ Phần II: Đúng/Sai", value=False)
        enable_part3 = st.checkbox("✅ Phần III: Điền số", value=False)
        
        st.divider()
        
        # Cấu hình số câu hỏi và điểm
        st.subheader("🔢 Số câu hỏi & Điểm")
        
        configs = {}
        
        if enable_part1:
            st.markdown("**Phần I:**")
            part1_questions = st.number_input(
                "Số câu ABCD:",
                min_value=1,
                max_value=100,
                value=40,
                step=1,
                key="part1_q"
            )
            part1_per_col = st.number_input(
                "Số câu/cột:",
                min_value=1,
                max_value=part1_questions,
                value=min(20, part1_questions),
                step=1,
                key="part1_col",
                help="Nếu phiếu có 2 cột, điền số câu trên mỗi cột"
            )
            part1_score = st.number_input(
                "Điểm Phần I:",
                min_value=0.0,
                max_value=100.0,
                value=5.0,
                step=0.5,
                key="part1_score"
            )
            configs['part1'] = {
                'total_questions': part1_questions,
                'questions_per_column': part1_per_col,
                'score': part1_score
            }
        
        if enable_part2:
            st.markdown("**Phần II:**")
            st.info("⚠️ **LƯU Ý QUAN TRỌNG**\nMỗi câu luôn có **4 ý (a,b,c,d)**\nĐiểm tính theo **số ý đúng**, không phải số câu!")
            part2_questions = st.number_input(
                "Số câu Đúng/Sai:",
                min_value=1,
                max_value=50,
                value=8,
                step=1,
                key="part2_q"
            )
            st.markdown(f"→ Tổng **{part2_questions * 4} ý** cần chấm")
            part2_score = st.number_input(
                "Điểm Phần II:",
                min_value=0.0,
                max_value=100.0,
                value=3.0,
                step=0.5,
                key="part2_score",
                help=f"Điểm được chia đều cho {part2_questions * 4} ý"
            )
            configs['part2'] = {
                'total_questions': part2_questions,
                'score': part2_score
            }
        
        if enable_part3:
            st.markdown("**Phần III:**")
            part3_questions = st.number_input(
                "Số câu điền số:",
                min_value=1,
                max_value=20,
                value=6,
                step=1,
                key="part3_q"
            )
            part3_score = st.number_input(
                "Điểm Phần III:",
                min_value=0.0,
                max_value=100.0,
                value=2.0,
                step=0.5,
                key="part3_score"
            )
            configs['part3'] = {
                'total_questions': part3_questions,
                'score': part3_score
            }
        
        # Tổng điểm
        total_score = sum(c['score'] for c in configs.values())
        st.divider()
        st.metric("**Tổng điểm:**", f"{total_score:.1f}")
        
        if total_score != 10.0:
            st.warning(f"⚠️ Tổng điểm là {total_score:.1f}, không phải 10 điểm!")
        
        st.divider()
        
        # Chế độ nâng cao
        st.subheader("🔧 Chế độ nâng cao")
        show_debug = st.checkbox("Hiển thị ảnh debug", value=False)
        calibration_mode = st.checkbox("Chế độ hiệu chuẩn (tìm tọa độ)", value=False)
        
        st.divider()
        
        # Hướng dẫn
        st.header("📖 Hướng Dẫn")
        st.markdown("""
        ### Các bước:
        1. **Chọn camera** (trước/sau)
        2. **Chọn phần** cần chấm
        3. **Nhập số câu & điểm**
        4. **Nhập đáp án** đúng
        5. **Chụp phiếu** bằng camera
        6. **Xem kết quả** ngay lập tức
        
        ### Lưu ý:
        - Đặt phiếu phẳng, rõ nét
        - Đủ ánh sáng, không bóng
        - Chụp vuông góc
        - **Phần II**: Điểm chia cho số **ý**, không phải số câu!
        """)
    
    # Main content
    if not (enable_part1 or enable_part2 or enable_part3):
        st.warning("⚠️ Vui lòng chọn ít nhất 1 phần để chấm!")
        return
    
    # === NHẬP ĐÁP ÁN ===
    st.header("1️⃣ Nhập Đáp Án Đúng")
    
    num_cols = sum([enable_part1, enable_part2, enable_part3])
    cols_input = st.columns(num_cols)
    
    part1_answers = None
    part2_answers = None
    part3_answers = None
    
    col_idx = 0
    
    # Phần I
    if enable_part1:
        with cols_input[col_idx]:
            st.subheader(f"Phần I: ABCD ({configs['part1']['total_questions']} câu)")
            part1_input = st.text_area(
                f"Nhập {configs['part1']['total_questions']} đáp án:",
                height=150,
                placeholder="ABCDABCD...\nhoặc xuống dòng",
                key="part1_input"
            )
            
            if part1_input.strip():
                part1_answers = parse_part1_answers(part1_input, configs['part1']['total_questions'])
                if part1_answers:
                    st.success(f"✅ Đã nhập đủ {len(part1_answers)} đáp án!")
                else:
                    st.error(f"❌ Lỗi: Cần đúng {configs['part1']['total_questions']} đáp án ABCD")
        col_idx += 1
    
    # Phần II
    if enable_part2:
        with cols_input[col_idx]:
            total_qs = configs['part2']['total_questions']
            total_items = total_qs * 4
            st.subheader(f"Phần II: Đúng/Sai ({total_qs} câu)")
            st.caption(f"⚠️ Tổng {total_items} ý (mỗi câu 4 ý)")
            part2_input = st.text_area(
                f"Mỗi câu 1 dòng (4 ý a,b,c,d):",
                height=150,
                placeholder="Đ S Đ Đ\nS Đ Đ S\n...",
                key="part2_input",
                help="Đ = Đúng, S = Sai\nMỗi dòng 4 giá trị"
            )
            
            if part2_input.strip():
                part2_answers = parse_part2_answers(part2_input, total_qs)
                if part2_answers:
                    st.success(f"✅ Đã nhập đủ {total_qs} câu ({total_items} ý)!")
                else:
                    st.error(f"❌ Lỗi: Cần đúng {total_qs} dòng, mỗi dòng 4 giá trị")
        col_idx += 1
    
    # Phần III
    if enable_part3:
        with cols_input[col_idx]:
            st.subheader(f"Phần III: Điền số ({configs['part3']['total_questions']} câu)")
            part3_input = st.text_area(
                "Mỗi câu 1 dòng:",
                height=150,
                placeholder="-1,5\n123\n0,75\n...",
                key="part3_input",
                help="Có thể có dấu âm, dấu phẩy"
            )
            
            if part3_input.strip():
                part3_answers = parse_part3_answers(part3_input, configs['part3']['total_questions'])
                if part3_answers:
                    st.success(f"✅ Đã nhập đủ {configs['part3']['total_questions']} câu!")
                else:
                    st.error(f"❌ Lỗi: Cần đúng {configs['part3']['total_questions']} dòng")
    
    st.divider()
    
    # === CHỤP PHIẾU ===
    st.header("2️⃣ Chụp Phiếu Trả Lời")
    
    # Kiểm tra đã nhập đáp án chưa
    ready_to_scan = (
        (not enable_part1 or part1_answers is not None) and
        (not enable_part2 or part2_answers is not None) and
        (not enable_part3 or part3_answers is not None)
    )
    
    if not ready_to_scan:
        st.warning("⚠️ Vui lòng nhập đầy đủ đáp án cho các phần đã chọn!")
        return
    
    # Camera input với config
    st.info(f"📷 Đang dùng: **{camera_option}**")
    
    # Streamlit camera_input không hỗ trợ chọn camera trực tiếp
    # Nhưng trên mobile, user có thể chọn camera trong dialog
    camera_photo = st.camera_input(
        "📷 Chụp phiếu:",
        key="camera_input",
        help="Trên điện thoại, bạn có thể chọn camera trước/sau khi nhấn chụp"
    )
    
    if camera_photo is None:
        st.info("💡 Nhấn nút camera để chụp phiếu trả lời")
        if use_back_camera:
            st.info("💡 **Tip**: Trên điện thoại, khi dialog xuất hiện, chọn camera sau để có chất lượng tốt hơn")
        return
    
    # === XỬ LÝ VÀ CHẤM ĐIỂM ===
    file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Hiển thị ảnh gốc
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image(camera_photo, caption="Ảnh gốc", use_container_width=True)
    
    # Xử lý ảnh
    with st.spinner("🔍 Đang phân tích phiếu..."):
        processed = preprocess_image(image)
        
        # Hiển thị ảnh đã xử lý (nếu bật debug)
        if show_debug:
            with col_img2:
                st.image(processed, caption="Ảnh đã xử lý", use_container_width=True, channels="GRAY")
        
        # Chấm từng phần
        results = {}
        
        # Phần I
        if enable_part1:
            student_part1 = []
            for q in range(configs['part1']['total_questions']):
                answer = detect_part1_answer(processed, q, configs['part1'])
                student_part1.append(answer)
            
            correct_count, details = grade_part1(student_part1, part1_answers, configs['part1'])
            results['part1'] = {
                'correct': correct_count,
                'total': configs['part1']['total_questions'],
                'score': configs['part1']['score'],
                'details': details
            }
        
        # Phần II
        if enable_part2:
            student_part2 = []
            items_per_q = PART2_LAYOUT['items_per_question']
            for q in range(configs['part2']['total_questions']):
                items = []
                for i in range(items_per_q):
                    answer = detect_part2_answer(processed, q, i, configs['part2'])
                    items.append(answer)
                student_part2.append(items)
            
            correct_count, details = grade_part2(student_part2, part2_answers, configs['part2'])
            total_items = configs['part2']['total_questions'] * items_per_q
            results['part2'] = {
                'correct': correct_count,
                'total': total_items,  # Tổng số ý, không phải số câu
                'score': configs['part2']['score'],
                'details': details
            }
        
        # Phần III
        if enable_part3:
            student_part3 = []
            for q in range(configs['part3']['total_questions']):
                answer = detect_part3_answer(processed, q, configs['part3'])
                student_part3.append(answer)
            
            correct_count, details = grade_part3(student_part3, part3_answers, configs['part3'])
            results['part3'] = {
                'correct': correct_count,
                'total': configs['part3']['total_questions'],
                'score': configs['part3']['score'],
                'details': details
            }
    
    # === HIỂN THỊ KẾT QUẢ ===
    st.divider()
    st.header("3️⃣ Kết Quả Chấm Điểm")
    
    # Tính điểm
    total_correct = 0
    total_questions = 0
    final_score = 0.0
    
    for part_key, part_result in results.items():
        total_correct += part_result['correct']
        total_questions += part_result['total']
        
        # Tính điểm từng phần
        part_score_per_question = part_result['score'] / part_result['total']
        part_earned_score = part_result['correct'] * part_score_per_question
        final_score += part_earned_score
    
    # Metrics tổng quan
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("📊 Tổng số đúng", f"{total_correct}/{total_questions}")
    with metric_cols[1]:
        st.metric("⭐ Điểm số", f"{final_score:.2f}/{total_score:.1f}")
    with metric_cols[2]:
        percent = (total_correct / total_questions) * 100 if total_questions > 0 else 0
        st.metric("📈 Tỷ lệ đúng", f"{percent:.1f}%")
    with metric_cols[3]:
        # Xếp loại dựa trên tỷ lệ % so với điểm tối đa
        score_percent = (final_score / total_score) * 100 if total_score > 0 else 0
        if score_percent >= 80:
            st.metric("🏆 Xếp loại", "Giỏi", delta="Tuyệt vời!")
        elif score_percent >= 65:
            st.metric("🏆 Xếp loại", "Khá", delta="Tốt!")
        elif score_percent >= 50:
            st.metric("🏆 Xếp loại", "Trung bình", delta="")
        else:
            st.metric("🏆 Xếp loại", "Yếu", delta="Cần cố gắng!")
    
    # Chi tiết từng phần
    tab_names = []
    if enable_part1:
        tab_names.append(f"Phần I ({results['part1']['correct']}/{results['part1']['total']}) - {results['part1']['correct'] * results['part1']['score'] / results['part1']['total']:.2f}đ")
    if enable_part2:
        tab_names.append(f"Phần II ({results['part2']['correct']}/{results['part2']['total']}) - {results['part2']['correct'] * results['part2']['score'] / results['part2']['total']:.2f}đ")
    if enable_part3:
        tab_names.append(f"Phần III ({results['part3']['correct']}/{results['part3']['total']}) - {results['part3']['correct'] * results['part3']['score'] / results['part3']['total']:.2f}đ")
    
    tabs = st.tabs(tab_names)
    
    tab_index = 0
    
    # Phần I
    if enable_part1:
        with tabs[tab_index]:
            st.subheader("📋 Chi Tiết Phần I")
            details = results['part1']['details']
            
            # Hiển thị dạng bảng 10 câu/hàng
            for i in range(0, len(details), 10):
                cols = st.columns(10)
                for j in range(10):
                    idx = i + j
                    if idx < len(details):
                        d = details[idx]
                        with cols[j]:
                            if d['is_correct']:
                                st.markdown(f"**{d['question']}** ✅")
                                st.success(d['student_answer'], icon="✅")
                            else:
                                st.markdown(f"**{d['question']}** ❌")
                                st.error(f"{d['student_answer']}\n→ {d['correct_answer']}", icon="❌")
            
            # Câu sai
            wrong = [d for d in details if not d['is_correct']]
            if wrong:
                st.divider()
                st.error(f"**Câu sai ({len(wrong)} câu):** " + ", ".join([str(d['question']) for d in wrong]))
        
        tab_index += 1
    
    # Phần II
    if enable_part2:
        with tabs[tab_index]:
            st.subheader("📋 Chi Tiết Phần II")
            st.info(f"⚠️ **Lưu ý**: Điểm tính theo số **ý đúng** ({results['part2']['total']} ý), không phải số câu!")
            
            details = results['part2']['details']
            
            # Hiển thị theo câu
            items_per_q = PART2_LAYOUT['items_per_question']
            for q in range(configs['part2']['total_questions']):
                st.markdown(f"**Câu {q+1}:**")
                cols = st.columns(items_per_q)
                for i in range(items_per_q):
                    idx = q * items_per_q + i
                    d = details[idx]
                    with cols[i]:
                        if d['is_correct']:
                            st.success(f"{d['question']}: {d['student_answer']}", icon="✅")
                        else:
                            st.error(f"{d['question']}: {d['student_answer']} → {d['correct_answer']}", icon="❌")
            
            # Ý sai
            wrong = [d for d in details if not d['is_correct']]
            if wrong:
                st.divider()
                st.error(f"**Ý sai ({len(wrong)} ý):** " + ", ".join([d['question'] for d in wrong]))
        
        tab_index += 1
    
    # Phần III
    if enable_part3:
        with tabs[tab_index]:
            st.subheader("📋 Chi Tiết Phần III")
            details = results['part3']['details']
            
            for d in details:
                if d['is_correct']:
                    st.success(f"{d['question']}: {d['student_answer']}", icon="✅")
                else:
                    st.error(f"{d['question']}: {d['student_answer']} → Đúng là: {d['correct_answer']}", icon="❌")
            
            # Câu sai
            wrong = [d for d in details if not d['is_correct']]
            if wrong:
                st.divider()
                st.error(f"**Câu sai ({len(wrong)} câu):** " + ", ".join([d['question'] for d in wrong]))
    
    # Chúc mừng nếu điểm cao
    if percent == 100:
        st.balloons()
        st.success("🎉 **HOÀN HẢO! 100% CHÍNH XÁC!**")
    elif percent >= 90:
        st.balloons()
        st.success("🎊 **TỐT LẮM! Gần hoàn hảo rồi!**")
    
    # === CALIBRATION MODE ===
    if calibration_mode:
        st.divider()
        st.header("🔧 Chế Độ Hiệu Chuẩn")
        st.markdown("""
        **Hướng dẫn tìm tọa độ:**
        1. Chụp 1 phiếu mẫu
        2. Tải ảnh về máy
        3. Mở bằng Paint/Photoshop
        4. Di chuột đến các ô cần tìm:
           - Phần I: Ô A câu 1
           - Phần II: Ô "Đúng" của Câu 1a
           - Phần III: Ô số 0 vị trí đầu tiên của Câu 1
        5. Ghi lại tọa độ (X, Y)
        6. Cập nhật vào code (PART1_LAYOUT, PART2_LAYOUT, PART3_LAYOUT)
        
        **Tham số cần chỉnh:**
        """)
        
        st.code(f"""
# PHẦN I
START_X = {PART1_LAYOUT['start_x']}
START_Y = {PART1_LAYOUT['start_y']}
BUBBLE_WIDTH = {PART1_LAYOUT['bubble_width']}
BUBBLE_HEIGHT = {PART1_LAYOUT['bubble_height']}
COL_GAP = {PART1_LAYOUT['col_gap']}
ROW_GAP = {PART1_LAYOUT['row_gap']}

# PHẦN II
START_X = {PART2_LAYOUT['start_x']}
START_Y = {PART2_LAYOUT['start_y']}
...

# PHẦN III
START_X = {PART3_LAYOUT['start_x']}
START_Y = {PART3_LAYOUT['start_y']}
...
        """, language="python")
        
        st.warning("⚠️ Sau khi chỉnh tham số, **khởi động lại app** để áp dụng thay đổi!")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <small>
        💡 <b>Lưu ý:</b> Nếu kết quả không chính xác, hãy bật "Chế độ hiệu chuẩn" 
        để tìm tọa độ chính xác cho phiếu của bạn.
        <br>
        📧 Hỗ trợ: NgânMiu.Store | STK | Version 2.1
        </small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    st.set_page_config(
        page_title="Chấm Trắc Nghiệm - Phiếu Chuẩn Bộ GD&ĐT",
        page_icon="📝",
        layout="wide"
    )
    
    # Header
    st.title("📝 CHẤM TRẮC NGHIỆM TỰ ĐỘNG")
    st.markdown("**Phiếu chuẩn Bộ GD&ĐT - Hỗ trợ đầy đủ 3 phần**")
    st.divider()
    
    # Sidebar - Cấu hình
    with st.sidebar:
        st.header("⚙️ Cấu Hình")
        
        # Chọn phần cần chấm
        st.subheader("📚 Chọn phần cần chấm")
        enable_part1 = st.checkbox("✅ Phần I: Trắc nghiệm ABCD (40 câu)", value=True)
        enable_part2 = st.checkbox("✅ Phần II: Đúng/Sai (8 câu x 4 ý)", value=False)
        enable_part3 = st.checkbox("✅ Phần III: Điền số (6 câu)", value=False)
        
        st.divider()
        
        # Chế độ nâng cao
        st.subheader("🔧 Chế độ nâng cao")
        show_debug = st.checkbox("Hiển thị ảnh debug", value=False)
        calibration_mode = st.checkbox("Chế độ hiệu chuẩn (tìm tọa độ)", value=False)
        
        st.divider()
        
        # Hướng dẫn
        st.header("📖 Hướng Dẫn")
        st.markdown("""
        ### Các bước:
        1. **Chọn phần** cần chấm
        2. **Nhập đáp án** đúng
        3. **Chụp phiếu** bằng camera
        4. **Xem kết quả** ngay lập tức
        
        ### Lưu ý:
        - Đặt phiếu phẳng, rõ nét
        - Đủ ánh sáng, không bóng
        - Chụp vuông góc
        
        ### Nếu kết quả sai:
        - Bật "Hiển thị ảnh debug"
        - Bật "Chế độ hiệu chuẩn"
        - Điều chỉnh tham số trong code
        """)
    
    # Main content
    if not (enable_part1 or enable_part2 or enable_part3):
        st.warning("⚠️ Vui lòng chọn ít nhất 1 phần để chấm!")
        return
    
    # === NHẬP ĐÁP ÁN ===
    st.header("1️⃣ Nhập Đáp Án Đúng")
    
    col_input1, col_input2, col_input3 = st.columns(3)
    
    part1_answers = None
    part2_answers = None
    part3_answers = None
    
    # Phần I
    if enable_part1:
        with col_input1:
            st.subheader("Phần I: ABCD (40 câu)")
            part1_input = st.text_area(
                "Nhập 40 đáp án:",
                height=150,
                placeholder="ABCDABCD...\nhoặc xuống dòng",
                key="part1_input"
            )
            
            if part1_input.strip():
                part1_answers = parse_part1_answers(part1_input)
                if part1_answers:
                    st.success(f"✅ Đã nhập đủ {len(part1_answers)} đáp án!")
                else:
                    st.error(f"❌ Lỗi: Cần đúng 40 đáp án ABCD")
    
    # Phần II
    if enable_part2:
        with col_input2:
            st.subheader("Phần II: Đúng/Sai (8 câu)")
            part2_input = st.text_area(
                "Mỗi câu 1 dòng (4 ý a,b,c,d):",
                height=150,
                placeholder="Đ S Đ Đ\nS Đ Đ S\n...",
                key="part2_input",
                help="Đ = Đúng, S = Sai\nMỗi dòng 4 giá trị"
            )
            
            if part2_input.strip():
                part2_answers = parse_part2_answers(part2_input)
                if part2_answers:
                    st.success(f"✅ Đã nhập đủ 8 câu!")
                else:
                    st.error(f"❌ Lỗi: Cần đúng 8 dòng, mỗi dòng 4 giá trị")
    
    # Phần III
    if enable_part3:
        with col_input3:
            st.subheader("Phần III: Điền số (6 câu)")
            part3_input = st.text_area(
                "Mỗi câu 1 dòng:",
                height=150,
                placeholder="-1,5\n123\n0,75\n...",
                key="part3_input",
                help="Có thể có dấu âm, dấu phẩy"
            )
            
            if part3_input.strip():
                part3_answers = parse_part3_answers(part3_input)
                if part3_answers:
                    st.success(f"✅ Đã nhập đủ 6 câu!")
                else:
                    st.error(f"❌ Lỗi: Cần đúng 6 dòng")
    
    st.divider()
    
    # === CHỤP PHIẾU ===
    st.header("2️⃣ Chụp Phiếu Trả Lời")
    
    # Kiểm tra đã nhập đáp án chưa
    ready_to_scan = (
        (not enable_part1 or part1_answers is not None) and
        (not enable_part2 or part2_answers is not None) and
        (not enable_part3 or part3_answers is not None)
    )
    
    if not ready_to_scan:
        st.warning("⚠️ Vui lòng nhập đầy đủ đáp án cho các phần đã chọn!")
        return
    
    # Camera
    camera_photo = st.camera_input("📷 Chụp phiếu:")
    
    if camera_photo is None:
        st.info("💡 Nhấn nút camera để chụp phiếu trả lời")
        return
    
    # === XỬ LÝ VÀ CHẤM ĐIỂM ===
    file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Hiển thị ảnh gốc
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image(camera_photo, caption="Ảnh gốc", use_container_width=True)
    
    # Xử lý ảnh
    with st.spinner("🔍 Đang phân tích phiếu..."):
        processed = preprocess_image(image)
        
        # Hiển thị ảnh đã xử lý (nếu bật debug)
        if show_debug:
            with col_img2:
                st.image(processed, caption="Ảnh đã xử lý", use_container_width=True, channels="GRAY")
        
        # Chấm từng phần
        results = {}
        
        # Phần I
        if enable_part1:
            student_part1 = []
            for q in range(PART1_CONFIG['total_questions']):
                answer = detect_part1_answer(processed, q)
                student_part1.append(answer)
            
            correct_count, details = grade_part1(student_part1, part1_answers)
            results['part1'] = {
                'correct': correct_count,
                'total': PART1_CONFIG['total_questions'],
                'details': details
            }
        
        # Phần II
        if enable_part2:
            student_part2 = []
            for q in range(PART2_CONFIG['total_questions']):
                items = []
                for i in range(PART2_CONFIG['items_per_question']):
                    answer = detect_part2_answer(processed, q, i)
                    items.append(answer)
                student_part2.append(items)
            
            correct_count, details = grade_part2(student_part2, part2_answers)
            results['part2'] = {
                'correct': correct_count,
                'total': PART2_CONFIG['total_questions'] * PART2_CONFIG['items_per_question'],
                'details': details
            }
        
        # Phần III
        if enable_part3:
            student_part3 = []
            for q in range(PART3_CONFIG['total_questions']):
                answer = detect_part3_answer(processed, q)
                student_part3.append(answer)
            
            correct_count, details = grade_part3(student_part3, part3_answers)
            results['part3'] = {
                'correct': correct_count,
                'total': PART3_CONFIG['total_questions'],
                'details': details
            }
    
    # === HIỂN THỊ KẾT QUẢ ===
    st.divider()
    st.header("3️⃣ Kết Quả Chấm Điểm")
    
    # Tổng hợp
    total_correct = sum(r['correct'] for r in results.values())
    total_questions = sum(r['total'] for r in results.values())
    score = (total_correct / total_questions) * MAX_SCORE if total_questions > 0 else 0
    
    # Metrics tổng quan
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("📊 Tổng số đúng", f"{total_correct}/{total_questions}")
    with metric_cols[1]:
        st.metric("⭐ Điểm số", f"{score:.2f}/{MAX_SCORE}")
    with metric_cols[2]:
        percent = (total_correct / total_questions) * 100 if total_questions > 0 else 0
        st.metric("📈 Tỷ lệ đúng", f"{percent:.1f}%")
    with metric_cols[3]:
        if percent >= 80:
            st.metric("🏆 Xếp loại", "Giỏi", delta="Tuyệt vời!")
        elif percent >= 65:
            st.metric("🏆 Xếp loại", "Khá", delta="Tốt!")
        elif percent >= 50:
            st.metric("🏆 Xếp loại", "Trung bình", delta="")
        else:
            st.metric("🏆 Xếp loại", "Yếu", delta="Cần cố gắng!")
    
    # Chi tiết từng phần
    tabs = st.tabs([
        f"Phần I ({results['part1']['correct']}/{results['part1']['total']})" if enable_part1 else None,
        f"Phần II ({results['part2']['correct']}/{results['part2']['total']})" if enable_part2 else None,
        f"Phần III ({results['part3']['correct']}/{results['part3']['total']})" if enable_part3 else None,
    ])
    
    tab_index = 0
    
    # Phần I
    if enable_part1:
        with tabs[tab_index]:
            st.subheader("📋 Chi Tiết Phần I")
            details = results['part1']['details']
            
            # Hiển thị dạng bảng 10 câu/hàng
            for i in range(0, len(details), 10):
                cols = st.columns(10)
                for j in range(10):
                    idx = i + j
                    if idx < len(details):
                        d = details[idx]
                        with cols[j]:
                            if d['is_correct']:
                                st.markdown(f"**{d['question']}** ✅")
                                st.success(d['student_answer'], icon="✅")
                            else:
                                st.markdown(f"**{d['question']}** ❌")
                                st.error(f"{d['student_answer']}\n→ {d['correct_answer']}", icon="❌")
            
            # Câu sai
            wrong = [d for d in details if not d['is_correct']]
            if wrong:
                st.divider()
                st.error(f"**Câu sai ({len(wrong)} câu):** " + ", ".join([str(d['question']) for d in wrong]))
        
        tab_index += 1
    
    # Phần II
    if enable_part2:
        with tabs[tab_index]:
            st.subheader("📋 Chi Tiết Phần II")
            details = results['part2']['details']
            
            # Hiển thị theo câu
            for q in range(PART2_CONFIG['total_questions']):
                st.markdown(f"**Câu {q+1}:**")
                cols = st.columns(4)
                for i in range(4):
                    idx = q * 4 + i
                    d = details[idx]
                    with cols[i]:
                        if d['is_correct']:
                            st.success(f"{d['question']}: {d['student_answer']}", icon="✅")
                        else:
                            st.error(f"{d['question']}: {d['student_answer']} → {d['correct_answer']}", icon="❌")
            
            # Ý sai
            wrong = [d for d in details if not d['is_correct']]
            if wrong:
                st.divider()
                st.error(f"**Ý sai ({len(wrong)} ý):** " + ", ".join([d['question'] for d in wrong]))
        
        tab_index += 1
    
    # Phần III
    if enable_part3:
        with tabs[tab_index]:
            st.subheader("📋 Chi Tiết Phần III")
            details = results['part3']['details']
            
            for d in details:
                if d['is_correct']:
                    st.success(f"{d['question']}: {d['student_answer']}", icon="✅")
                else:
                    st.error(f"{d['question']}: {d['student_answer']} → Đúng là: {d['correct_answer']}", icon="❌")
            
            # Câu sai
            wrong = [d for d in details if not d['is_correct']]
            if wrong:
                st.divider()
                st.error(f"**Câu sai ({len(wrong)} câu):** " + ", ".join([d['question'] for d in wrong]))
    
    # Chúc mừng nếu điểm cao
    if percent == 100:
        st.balloons()
        st.success("🎉 **HOÀN HẢO! 100% CHÍNH XÁC!**")
    elif percent >= 90:
        st.balloons()
        st.success("🎊 **TỐT LẮM! Gần hoàn hảo rồi!**")
    
    # === CALIBRATION MODE ===
    if calibration_mode:
        st.divider()
        st.header("🔧 Chế Độ Hiệu Chuẩn")
        st.markdown("""
        **Hướng dẫn tìm tọa độ:**
        1. Chụp 1 phiếu mẫu
        2. Tải ảnh về máy
        3. Mở bằng Paint/Photoshop
        4. Di chuột đến các ô cần tìm:
           - Phần I: Ô A câu 1
           - Phần II: Ô "Đúng" của Câu 1a
           - Phần III: Ô số 0 vị trí đầu tiên của Câu 1
        5. Ghi lại tọa độ (X, Y)
        6. Cập nhật vào code (PART1_CONFIG, PART2_CONFIG, PART3_CONFIG)
        
        **Tham số cần chỉnh:**
        """)
        
        st.code(f"""
# PHẦN I
START_X = {PART1_CONFIG['start_x']}
START_Y = {PART1_CONFIG['start_y']}
BUBBLE_WIDTH = {PART1_CONFIG['bubble_width']}
BUBBLE_HEIGHT = {PART1_CONFIG['bubble_height']}
COL_GAP = {PART1_CONFIG['col_gap']}
ROW_GAP = {PART1_CONFIG['row_gap']}

# PHẦN II
START_X = {PART2_CONFIG['start_x']}
START_Y = {PART2_CONFIG['start_y']}
...

# PHẦN III
START_X = {PART3_CONFIG['start_x']}
START_Y = {PART3_CONFIG['start_y']}
...
        """, language="python")
        
        st.warning("⚠️ Sau khi chỉnh tham số, **khởi động lại app** để áp dụng thay đổi!")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <small>
        💡 <b>Lưu ý:</b> Nếu kết quả không chính xác, hãy bật "Chế độ hiệu chuẩn" 
        để tìm tọa độ chính xác cho phiếu của bạn.
        <br>
        📧 Hỗ trợ: NgânMiu.Store | STK
        </small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
