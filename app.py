import streamlit as st
from PIL import Image
import time
import hashlib

from processing.intensity import negative, log_transform, gamma_correction, piecewise_linear
from processing.histogram import hist_equalization, clahe_equalization, ahe_equalization, ahe_equalization_fast
from processing.applications import enhance_license_plate, enhance_satellite_image, enhance_low_light_image
from utils.image_io import pil_to_np, np_to_pil
from utils.plot import plot_histogram

# Cấu hình Streamlit cơ bản
st.set_page_config(
    page_title="Xử lý ảnh - Tiểu luận 1",
    page_icon="📸",
    initial_sidebar_state="collapsed"
)

# Cache để tăng tốc xử lý
@st.cache_data(show_spinner=False, max_entries=10)
def cached_image_processing(img_hash, method, params):
    """Cache kết quả xử lý ảnh để tránh tính toán lại"""
    return None  # Placeholder - sẽ được override bởi logic thực tế

def get_image_hash(img_array):
    """Tạo hash cho ảnh để cache"""
    return hashlib.md5(img_array.tobytes()).hexdigest()[:8]

st.title("Xử lý ảnh - Tiểu luận 1")

uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img = pil_to_np(image)
    
    tab1, tab2, tab3 = st.tabs(["Biến đổi cường độ sáng", "Cân bằng histogram", "Ứng dụng thực tế"])
    
    with tab1:
        method = st.selectbox(
            "Chọn phương pháp cường độ sáng",
            ["Negative", "Log", "Gamma/Power-law", "Piecewise-linear"],
        )

        # Xử lý các phương pháp biến đổi cơ bản
        import cv2
        if method == "Gamma/Power-law":
            st.caption("Power-law transformation: s = c * r^γ")
            col_c, col_g = st.columns(2)
            with col_c:
                c_val = st.slider("Hằng số c", 0.1, 3.0, 1.0, 0.1)
            with col_g:
                gamma_val = st.slider("Tham số γ (gamma)", 0.1, 3.0, 1.0, 0.1)
            processed = gamma_correction(img, gamma_val, c_val)
            display_original = image
        elif method == "Piecewise-linear":
            st.caption("Biến đổi tuyến tính từng đoạn (contrast stretching)")
            col_a, col_b = st.columns(2)
            with col_a:
                r1 = st.slider("r1", 0, 255, 50)
                r2 = st.slider("r2", 0, 255, 200)
            with col_b:
                s1 = st.slider("s1", 0, 255, 20)
                s2 = st.slider("s2", 0, 255, 230)
            # Đảm bảo r2 > r1 hợp lệ (hàm xử lý cũng tự bảo vệ)
            processed = piecewise_linear(img, r1, s1, r2, s2)
            display_original = image
        elif method == "Negative":
            processed = negative(img)
            display_original = image
        elif method == "Log":
            st.caption("Log transformation: s = c * log(1 + r)")
            c_val = st.slider("Hằng số c (range rộng để thấy rõ khác biệt)", 0.1, 50.0, 1.0, 0.1)
            processed = log_transform(img, c_val)
            display_original = image

        # Hiển thị ảnh gốc và ảnh sau biến đổi song song nhau
        st.subheader("So sánh kết quả")
        col1, col2 = st.columns(2)
        with col1:
            st.image(display_original, caption="Ảnh gốc", use_container_width=True)
        with col2:
            st.image(processed, caption=f"Ảnh sau {method}", use_container_width=True)
            
        # Hiển thị histogram gốc và histogram sau biến đổi song song
        st.subheader("So sánh histogram")
        col1, col2 = st.columns(2)
        with col1:
            st.image(plot_histogram(img), caption="Histogram gốc", use_container_width=True)
        with col2:
            st.image(plot_histogram(processed), caption=f"Histogram sau {method}", use_container_width=True)
            
        # Tạo phần tải xuống ở giữa màn hình
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Tải xuống ảnh kết quả
            result_pil = np_to_pil(processed)
            st.download_button("📥 Tải ảnh kết quả", 
                            data=result_pil.tobytes(),
                            file_name=f"result_{method}.png",
                            mime="image/png",
                            key="download_intensity")
    with tab2:
        st.caption("Cân bằng lược đồ mức xám (Histogram Equalization / AHE / CLAHE)")
        he_method = st.selectbox("Chọn phương pháp cân bằng", ["Histogram Equalization", "AHE", "CLAHE"]) 
        import cv2
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        if he_method == "Histogram Equalization":
            processed_he = hist_equalization(gray_img)
        elif he_method == "AHE":
            st.info("📊 AHE với parameters tự động tối ưu dựa trên đặc điểm ảnh")
            
            # Tùy chọn manual override
            manual_params = st.checkbox("🔧 Tùy chỉnh parameters thủ công", value=False)
            
            if manual_params:
                col_win, col_fast = st.columns(2)
                with col_win:
                    window = st.slider("Window Size", 16, 128, 64, 16)
                with col_fast:
                    step_size = st.slider("Step Size (tăng để nhanh hơn)", 4, 16, 8, 2)
                
                processed_he = ahe_equalization_fast(gray_img, window, step_size)
            else:
                # Sử dụng auto parameters
                st.success("✅ Sử dụng parameters tự động tối ưu")
                processed_he = ahe_equalization_fast(gray_img)
        else:  # CLAHE
            clip = st.slider("Clip Limit", 1.0, 5.0, 2.0, 0.1)
            grid = st.slider("Tile Grid Size", 4, 16, 8, 1)
            processed_he = clahe_equalization(gray_img, clip, grid)

        # So sánh ảnh gốc (xám) và ảnh sau HE/CLAHE
        st.subheader("So sánh kết quả")
        c1, c2 = st.columns(2)
        with c1:
            st.image(gray_img, caption="Ảnh gốc (grayscale)", use_container_width=True)
        with c2:
            st.image(processed_he, caption=f"Ảnh sau {he_method}", use_container_width=True)

        st.subheader("So sánh histogram")
        c1, c2 = st.columns(2)
        with c1:
            st.image(plot_histogram(gray_img), caption="Histogram gốc", use_container_width=True)
        with c2:
            st.image(plot_histogram(processed_he), caption=f"Histogram sau {he_method}", use_container_width=True)

        # Nút tải xuống
        d1, d2, d3 = st.columns([1, 2, 1])
        with d2:
            result_pil = np_to_pil(processed_he)
            st.download_button(
                "📥 Tải ảnh kết quả",
                data=result_pil.tobytes(),
                file_name=f"result_{he_method}.png",
                mime="image/png",
                key="download_hist",
            )
    with tab3:
        application = st.selectbox(
            "Chọn ứng dụng thực tế",
            ["Xử lý biển số xe", "Cải thiện ảnh vệ tinh", "Xử lý ảnh ánh sáng kém"]
        )
        
        # Hiển thị progress bar khi xử lý
        progress_container = st.empty()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                if application == "Xử lý biển số xe":
                    st.info("Tiền xử lý ảnh biển số xe để tối ưu cho nhận dạng ký tự (OCR). Kết quả là ảnh nhị phân với ký tự rõ nét.")
                    
                    status_text.text("Processing license plate...")
                    progress_bar.progress(30)
                    processed = enhance_license_plate(img)
                    
                elif application == "Cải thiện ảnh vệ tinh":
                    st.info("Cải thiện chất lượng ảnh vệ tinh để hỗ trợ phân tích trong các hệ thống thông tin địa lý (GIS).")
                    
                    status_text.text("Processing satellite image...")
                    progress_bar.progress(30)
                    processed = enhance_satellite_image(img)
                    
                elif application == "Xử lý ảnh ánh sáng kém":
                    st.info("🌙 Nâng cao chất lượng ảnh chụp trong điều kiện ánh sáng kém. Sử dụng HSV color space để bảo toàn màu sắc tự nhiên và tránh nhiễu màu.")
                    
                    status_text.text("Processing low-light image...")
                    progress_bar.progress(30)
                    processed = enhance_low_light_image(img)
                
                progress_bar.progress(70)
                status_text.text("Preparing output...")
                
                progress_bar.progress(100)
                status_text.text("✅ Processing completed!")
                
                # Clear progress after delay
                time.sleep(0.5)
                progress_container.empty()
                
            except Exception as e:
                progress_container.empty()
                st.error(f"Error during processing: {str(e)}")
                st.error("Please try with a different image.")

        # Bảo đảm ảnh hiển thị luôn hợp lệ (kể cả khi là ảnh xám/nhị phân)
        import cv2
        import numpy as np
        if processed is None:
            processed_safe = img
        else:
            processed_safe = np.array(processed)
            if processed_safe.dtype != np.uint8:
                processed_safe = np.clip(processed_safe, 0, 255).astype(np.uint8)
        # For display, convert gray => RGB to avoid theme quirks
        processed_vis = (
            cv2.cvtColor(processed_safe, cv2.COLOR_GRAY2RGB)
            if processed_safe.ndim == 2
            else processed_safe
        )

        st.subheader("So sánh kết quả")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Ảnh gốc", use_container_width=True)
        with col2:
            st.image(processed_vis, caption=f"Ảnh sau khi xử lý ({application})", use_container_width=True)
            
        st.subheader("So sánh histogram")
        col1, col2 = st.columns(2)
        with col1:
            st.image(plot_histogram(img), caption="Histogram gốc", use_container_width=True)
        with col2:
            st.image(plot_histogram(processed_safe), caption="Histogram sau xử lý", use_container_width=True)
        
        # Tạo phần tải xuống ở giữa màn hình
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            result_pil = np_to_pil(processed_safe)
            st.download_button("📥 Tải ảnh kết quả", 
                            data=result_pil.tobytes(),
                            file_name=f"result_{application}.jpg",
                            mime="image/jpeg",
                            key="download_application")
