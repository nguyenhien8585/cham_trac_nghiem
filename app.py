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

# === PHẦN I: Trắc nghiệm ABCD (40 câu) ===
PART1_CONFIG = {
    'total_questions': 40,
    'questions_per_column': 20,  # 2 cột: 1-20 và 21-40
    'start_x': 50,              # Tọa độ X bắt đầu cột 1
    'start_y': 180,             # Tọa độ Y bắt đầu câu 1
    'bubble_width': 28,         # Chiều rộng ô tròn
    'bubble_height': 28,        # Chiều cao ô tròn
    'col_gap': 35,              # Khoảng cách A→B→C→D
    'row_gap': 35,              # Khoảng cách câu 1→2→3
    'column_spacing': 300,      # Khoảng cách cột 1→cột 2
}

# === PHẦN II: Đúng/Sai (8 câu x 4 ý) ===
PART2_CONFIG = {
    'total_questions': 8,       # 8 câu (Câu 1-8)
    'items_per_question': 4,    # Mỗi câu có 4 ý (a,b,c,d)
    'start_x': 600,             # Tọa độ X bắt đầu Câu 1
    'start_y': 180,             # Tọa độ Y bắt đầu
    'bubble_width': 20,         # Kích thước ô Đúng/Sai
    'bubble_height': 20,
    'col_gap': 35,              # Khoảng cách Đúng→Sai
    'row_gap': 25,              # Khoảng cách a→b→c→d
    'question_gap': 140,        # Khoảng cách Câu 1→Câu 2
}

# === PHẦN III: Điền số (6 câu) ===
PART3_CONFIG = {
    'total_questions': 6,       # 6 câu điền số
    'digits_per_question': 6,   # Mỗi câu tối đa 6 ký tự
    'digit_options': 11,        # 0-9 + dấu phẩy = 11 option
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

# Điểm tối đa
MAX_SCORE = 10.0

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

def detect_part1_answer(image: np.ndarray, question_num: int) -> Optional[str]:
    """
    Phát hiện đáp án ABCD cho một câu hỏi ở Phần I
    
    Args:
        image: Ảnh đã threshold
        question_num: Số thứ tự câu (0-39)
    
    Returns:
        Đáp án ('A', 'B', 'C', 'D') hoặc None
    """
    cfg = PART1_CONFIG
    options = ['A', 'B', 'C', 'D']
    fill_ratios = []
    
    # Xác định cột (0 hoặc 1)
    column = question_num // cfg['questions_per_column']
    row_in_column = question_num % cfg['questions_per_column']
    
    # Kiểm tra từng ô A, B, C, D
    for i in range(4):
        x = cfg['start_x'] + (i * cfg['col_gap']) + (column * cfg['column_spacing'])
        y = cfg['start_y'] + (row_in_column * cfg['row_gap'])
        
        bubble = extract_bubble(image, x, y, cfg['bubble_width'], cfg['bubble_height'])
        fill_ratio = is_bubble_filled(bubble)
        fill_ratios.append(fill_ratio)
    
    # Tìm ô có tỷ lệ đen cao nhất
    max_fill = max(fill_ratios)
    
    if max_fill >= FILL_THRESHOLD:
        max_index = fill_ratios.index(max_fill)
        return options[max_index]
    
    return None


def grade_part1(student_answers: List[Optional[str]], 
                correct_answers: List[str]) -> Tuple[int, List[dict]]:
    """
    Chấm điểm Phần I
    
    Args:
        student_answers: Đáp án học sinh
        correct_answers: Đáp án đúng
    
    Returns:
        Tuple (số câu đúng, chi tiết từng câu)
    """
    correct_count = 0
    details = []
    
    for i in range(PART1_CONFIG['total_questions']):
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

def detect_part2_answer(image: np.ndarray, question_num: int, item_index: int) -> Optional[str]:
    """
    Phát hiện đáp án Đúng/Sai cho một ý của câu hỏi ở Phần II
    
    Args:
        image: Ảnh đã threshold
        question_num: Số câu (0-7)
        item_index: Số ý (0=a, 1=b, 2=c, 3=d)
    
    Returns:
        'Đúng', 'Sai' hoặc None
    """
    cfg = PART2_CONFIG
    options = ['Đúng', 'Sai']
    fill_ratios = []
    
    # Kiểm tra ô Đúng và Sai
    for i in range(2):
        x = cfg['start_x'] + (i * cfg['col_gap'])
        y = cfg['start_y'] + (question_num * cfg['question_gap']) + (item_index * cfg['row_gap'])
        
        bubble = extract_bubble(image, x, y, cfg['bubble_width'], cfg['bubble_height'])
        fill_ratio = is_bubble_filled(bubble)
        fill_ratios.append(fill_ratio)
    
    max_fill = max(fill_ratios)
    
    if max_fill >= FILL_THRESHOLD:
        max_index = fill_ratios.index(max_fill)
        return options[max_index]
    
    return None


def grade_part2(student_answers: List[List[Optional[str]]], 
                correct_answers: List[List[str]]) -> Tuple[int, List[dict]]:
    """
    Chấm điểm Phần II
    
    Args:
        student_answers: Đáp án học sinh [[a,b,c,d], [a,b,c,d], ...]
        correct_answers: Đáp án đúng
    
    Returns:
        Tuple (số ý đúng, chi tiết)
    """
    correct_count = 0
    details = []
    items_labels = ['a', 'b', 'c', 'd']
    
    for q in range(PART2_CONFIG['total_questions']):
        for i in range(PART2_CONFIG['items_per_question']):
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

def detect_part3_answer(image: np.ndarray, question_num: int) -> Optional[str]:
    """
    Phát hiện số điền vào cho một câu ở Phần III
    
    Args:
        image: Ảnh đã threshold
        question_num: Số câu (0-5)
    
    Returns:
        Chuỗi số (VD: "-1,5", "123") hoặc None
    """
    cfg = PART3_CONFIG
    digit_map = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ',']
    result = []
    
    # Duyệt qua 6 vị trí (từ trái sang phải)
    for pos in range(cfg['digits_per_question']):
        fill_ratios = []
        
        # Kiểm tra 11 option (0-9 + dấu phẩy)
        for digit_idx in range(cfg['digit_options']):
            x = cfg['start_x'] + (pos * cfg['col_gap'])
            y = cfg['start_y'] + (question_num * cfg['question_gap']) + (digit_idx * cfg['row_gap'])
            
            bubble = extract_bubble(image, x, y, cfg['bubble_width'], cfg['bubble_height'])
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
                correct_answers: List[str]) -> Tuple[int, List[dict]]:
    """
    Chấm điểm Phần III
    
    Args:
        student_answers: Đáp án học sinh
        correct_answers: Đáp án đúng
    
    Returns:
        Tuple (số câu đúng, chi tiết)
    """
    correct_count = 0
    details = []
    
    for i in range(PART3_CONFIG['total_questions']):
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

def parse_part1_answers(text: str) -> Optional[List[str]]:
    """
    Phân tích đáp án Phần I (ABCD)
    
    Args:
        text: Chuỗi đáp án (VD: "ABCDABCD...")
    
    Returns:
        List các đáp án hoặc None nếu lỗi
    """
    text = text.replace(' ', '').replace('\n', '').replace('\r', '').upper()
    answers = [c for c in text if c in 'ABCD']
    
    if len(answers) != PART1_CONFIG['total_questions']:
        return None
    
    return answers


def parse_part2_answers(text: str) -> Optional[List[List[str]]]:
    """
    Phân tích đáp án Phần II (Đúng/Sai)
    
    Args:
        text: Chuỗi đáp án, mỗi câu 1 dòng
        Format: "Đ S Đ Đ" hoặc "D S D D" (4 ý cho mỗi câu)
    
    Returns:
        List [[a,b,c,d], [a,b,c,d], ...] hoặc None nếu lỗi
    """
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    
    if len(lines) != PART2_CONFIG['total_questions']:
        return None
    
    answers = []
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
        
        if len(items) != PART2_CONFIG['items_per_question']:
            return None
        
        answers.append(items)
    
    return answers


def parse_part3_answers(text: str) -> Optional[List[str]]:
    """
    Phân tích đáp án Phần III (Điền số)
    
    Args:
        text: Chuỗi đáp án, mỗi câu 1 dòng
        Format: "-1,5" hoặc "123"
    
    Returns:
        List các số hoặc None nếu lỗi
    """
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    
    if len(lines) != PART3_CONFIG['total_questions']:
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
