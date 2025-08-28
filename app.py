import streamlit as st
from PIL import Image

from processing.intensity import negative, log_transform, gamma_correction, piecewise_linear
from processing.histogram import hist_equalization, clahe_equalization
from processing.applications import enhance_license_plate, enhance_satellite_image, enhance_low_light_image, restore_document_image
from utils.image_io import pil_to_np, np_to_pil
from utils.plot import plot_histogram

st.title("Xử lý ảnh - Tiểu luận 1")

uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img = pil_to_np(image)
    
    tab1, tab2, tab3 = st.tabs(["Biến đổi cường độ sáng", "Cân bằng histogram", "Ứng dụng thực tế"])
    
    with tab1:
        method = st.selectbox(
            "Chọn phương pháp cường độ sáng",
            ["Negative", "Log", "Gamma", "Piecewise-linear"],
        )

        # Xử lý các phương pháp biến đổi cơ bản
        import cv2
        if method == "Gamma":
            gamma_val = st.slider("Chọn Gamma", 0.1, 3.0, 1.0, 0.1)
            processed = gamma_correction(img, gamma_val)
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
            processed = log_transform(img)
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
        st.caption("Cân bằng lược đồ mức xám (Histogram Equalization / CLAHE)")
        he_method = st.selectbox("Chọn phương pháp cân bằng", ["Histogram Equalization", "CLAHE"]) 
        import cv2
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        if he_method == "Histogram Equalization":
            processed_he = hist_equalization(gray_img)
        else:
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
            ["Xử lý biển số xe", "Cải thiện ảnh vệ tinh", "Xử lý ảnh ánh sáng kém", "Khôi phục ảnh giấy tờ"]
        )
        
        if application == "Xử lý biển số xe":
            st.info("Tiền xử lý ảnh biển số xe để tối ưu cho nhận dạng ký tự (OCR). Kết quả là ảnh nhị phân với ký tự rõ nét.")
            
            processed = enhance_license_plate(img)
        elif application == "Cải thiện ảnh vệ tinh":
            st.info("Cải thiện chất lượng ảnh vệ tinh để hỗ trợ phân tích trong các hệ thống thông tin địa lý (GIS).")
            processed = enhance_satellite_image(img)
        elif application == "Xử lý ảnh ánh sáng kém":
            st.info("Nâng cao chất lượng ảnh chụp trong điều kiện ánh sáng kém, cải thiện độ sáng và độ tương phản.")
            processed = enhance_low_light_image(img)
        elif application == "Khôi phục ảnh giấy tờ":
            st.info("Khôi phục chất lượng ảnh giấy tờ cũ, mờ hoặc bị hư hỏng để đảm bảo tính rõ ràng của văn bản.")
            
            # Thêm tùy chọn kiểu output
            output_type = st.radio(
                "Chọn kiểu kết quả:",
                ["Ảnh xám", "Ảnh nhị phân"],
                index=0
            )
            
            binary_output = (output_type == "Ảnh nhị phân")
            processed = restore_document_image(img, binary_output)

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
                            file_name=f"result_{application}.png",
                            mime="image/png",
                            key="download_application")
