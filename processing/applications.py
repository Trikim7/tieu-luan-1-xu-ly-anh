import cv2
import numpy as np
from .intensity import negative, log_transform, gamma_correction, piecewise_linear
from .histogram import hist_equalization, clahe_equalization, ahe_equalization_fast

def adaptive_threshold_custom(img, max_value=255, block_size=21, C=8):
    """
    Adaptive thresholding tối ưu hóa với vectorization
    Time Complexity: O(H×W) với vectorized operations
    """
    h, w = img.shape
    
    # Giới hạn kích thước để tránh quá chậm
    if h * w > 2000000:  # > 2M pixels
        # Resize ảnh xuống kích thước hợp lý
        from PIL import Image
        scale = np.sqrt(2000000 / (h * w))
        new_h, new_w = int(h * scale), int(w * scale)
        img_small = np.array(Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS))
        
        # Xử lý ảnh nhỏ
        binary_small = adaptive_threshold_custom(img_small, max_value, 
                                               max(11, block_size // 2), C)
        
        # Resize lại về kích thước gốc
        binary = np.array(Image.fromarray(binary_small).resize((w, h), Image.NEAREST))
        return binary.astype(np.uint8)
    
    # Sử dụng cv2.boxFilter cho convolution nhanh
    try:
        import cv2
        # Box filter để tính mean local nhanh
        kernel = np.ones((block_size, block_size), np.float32) / (block_size * block_size)
        mean_img = cv2.filter2D(img.astype(np.float32), -1, kernel)
        
        # Vectorized thresholding
        binary = np.where(img > mean_img - C, max_value, 0).astype(np.uint8)
        
    except ImportError:
        # Fallback với scipy convolution nếu không có cv2
        from scipy.ndimage import uniform_filter
        mean_img = uniform_filter(img.astype(np.float32), size=block_size, mode='constant')
        binary = np.where(img > mean_img - C, max_value, 0).astype(np.uint8)
    
    return binary

def enhance_license_plate(img):
    """
    Tiền xử lý ảnh cho nhận dạng biển số xe
    """
    try:
        # Chuyển sang ảnh xám nếu là ảnh màu
        if len(img.shape) == 3:
            # Convert to grayscale manually
            gray = np.dot(img[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
        else:
            gray = img.copy()

        # Bước 1: Cải thiện độ tương phản bằng CLAHE
        enhanced = clahe_equalization(gray, clip=3.0, grid=8)
        
        # Bước 2: Gamma correction để cân bằng độ sáng
        enhanced = gamma_correction(enhanced, gamma=0.8)
        
        # Bước 3: Piecewise linear transformation để tăng contrast
        # Tìm percentile để xác định ngưỡng
        p5 = np.percentile(enhanced, 5)
        p95 = np.percentile(enhanced, 95)
        enhanced = piecewise_linear(enhanced, int(p5), 20, int(p95), 235)
        
        # Bước 4: Adaptive thresholding tự implement
        result = adaptive_threshold_custom(enhanced, 255, 21, 8)
        
        return result
    
    except Exception as e:
        print(f"Lỗi trong enhance_license_plate: {e}")
        # Fallback đơn giản
        if len(img.shape) == 3:
            gray_fallback = np.dot(img[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
        else:
            gray_fallback = img.copy()
        
        enhanced_fallback = clahe_equalization(gray_fallback, clip=2.0, grid=8)
        return adaptive_threshold_custom(enhanced_fallback, 255, 15, 5)

def enhance_satellite_image(img):
    """
    Cải thiện ảnh vệ tinh trong GIS
    """
    # Kiểm tra xem ảnh là màu hay xám
    is_color = len(img.shape) == 3
    
    if is_color:
        # Xử lý từng kênh màu riêng biệt
        channels = [img[:,:,i] for i in range(3)]
        enhanced_channels = []
        
        for channel in channels:
            # Bước 1: Log transformation để tăng cường vùng tối
            log_enhanced = log_transform(channel, c=1.2)
            
            # Bước 2: CLAHE để cải thiện local contrast
            clahe_enhanced = clahe_equalization(log_enhanced, clip=2.0, grid=12)
            
            # Bước 3: Gamma correction để điều chỉnh độ sáng tổng thể
            gamma_enhanced = gamma_correction(clahe_enhanced, gamma=0.9)
            
            # Bước 4: Piecewise linear để tăng contrast cuối
            p2 = np.percentile(gamma_enhanced, 2)
            p98 = np.percentile(gamma_enhanced, 98)
            final_enhanced = piecewise_linear(gamma_enhanced, int(p2), 10, int(p98), 245)
            
            enhanced_channels.append(final_enhanced)
        
        # Kết hợp các kênh
        enhanced = np.stack(enhanced_channels, axis=2)
        
    else:
        # Với ảnh xám
        # Bước 1: Log transformation
        log_enhanced = log_transform(img, c=1.2)
        
        # Bước 2: CLAHE
        clahe_enhanced = clahe_equalization(log_enhanced, clip=2.0, grid=12)
        
        # Bước 3: Gamma correction
        gamma_enhanced = gamma_correction(clahe_enhanced, gamma=0.9)
        
        # Bước 4: Contrast stretching
        p2 = np.percentile(gamma_enhanced, 2)
        p98 = np.percentile(gamma_enhanced, 98)
        enhanced = piecewise_linear(gamma_enhanced, int(p2), 10, int(p98), 245)
    
    return enhanced

def enhance_low_light_image(img, method="enhanced"):
    """
    Nâng cao chất lượng ảnh chụp trong điều kiện ánh sáng kém
    """
    # Chuyển sang HSV để xử lý riêng brightness và saturation
    import cv2
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    
    # Chỉ xử lý kênh V (brightness), giữ nguyên H để tránh color shift
    
    # Bước 1: Gentle gamma correction để tăng chi tiết vùng tối
    v_normalized = v / 255.0
    v_gamma = np.power(v_normalized, 0.8) * 255.0  # Gamma nhẹ nhàng hơn
    
    # Bước 1.5: Tăng brightness vừa phải 
    v_brightened = np.clip(v_gamma + 20, 0, 255)  # Chỉ tăng 20 đơn vị
    
    # Bước 2: CLAHE với parameters cân bằng
    v_clahe = clahe_equalization(v_brightened.astype(np.uint8), clip=2.0, grid=8)  # Tăng clip, giảm grid
    
    # Bước 2.5: Manual smoothing để giảm CLAHE artifacts (3x3 box filter)
    def manual_smooth(image):
        h, w = image.shape
        smoothed = image.copy().astype(np.float32)
        
        # Xử lý interior pixels (bỏ qua viền)
        for i in range(1, h-1):
            for j in range(1, w-1):
                # 3x3 box filter averaging
                neighborhood = image[i-1:i+2, j-1:j+2].astype(np.float32)
                smoothed[i, j] = np.mean(neighborhood)
        
        return smoothed.astype(np.uint8)
    
    v_clahe = manual_smooth(v_clahe)
    
    # Bước 3: Blend cân bằng giữa CLAHE và original
    v_enhanced = np.clip(
        0.7 * v_clahe.astype(np.float32) + 0.3 * v,  # 70/30 blend
        0, 255
    )
    
    # Bước 4: Giữ saturation tự nhiên
    s_reduced = np.clip(s * 0.95, 0, 255)  # Chỉ giảm nhẹ 5%
    
    # Kết hợp lại HSV và chuyển về RGB
    hsv_enhanced = np.stack([h, s_reduced, v_enhanced], axis=2).astype(np.uint8)
    enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)
    
    # Bước cuối: Gentle contrast stretching chỉ khi cần thiết
    mean_brightness = np.mean(enhanced)
    if mean_brightness < 80:  # Chỉ áp dụng khi ảnh còn quá tối
        p5 = np.percentile(enhanced, 5)  
        p95 = np.percentile(enhanced, 95)  
        if p95 - p5 < 100:  # Chỉ khi contrast thấp
            enhanced = piecewise_linear(enhanced, int(p5), 10, int(p95), 240)  # Gentle range [10,240]
    
    return enhanced


