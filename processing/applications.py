import cv2
import numpy as np
from .intensity import negative, log_transform, gamma_correction
from .histogram import hist_equalization, clahe_equalization

def enhance_license_plate(img, processing_mode="enhanced_grayscale"):
    """
    Tiền xử lý ảnh cho nhận dạng biển số xe
    Chỉ có 2 chế độ đơn giản: enhanced_grayscale và binary
    """
    try:
        # Chuyển sang ảnh xám nếu là ảnh màu
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()

        # Cải thiện độ tương phản và giảm nhiễu
        enhanced = clahe_equalization(gray, clip=2.5, grid=8)
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

        if processing_mode == "binary":
            # Chế độ nhị phân cho OCR
            result = cv2.adaptiveThreshold(enhanced, 255, 
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 
                                           21, 8)
            return result
        else:
            # Chế độ mặc định: ảnh xám cải thiện
            kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) / 1.0
            sharpened = cv2.filter2D(enhanced, -1, kernel_sharp)
            result = np.clip(sharpened, 0, 255).astype(np.uint8)
            return result
    
    except Exception as e:
        print(f"Lỗi trong enhance_license_plate: {e}")
        # Fallback: trả về ảnh xám cải thiện đơn giản
        if len(img.shape) == 3:
            gray_fallback = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray_fallback = img.copy()
        
        enhanced_fallback = clahe_equalization(gray_fallback, clip=2.0, grid=8)
        return enhanced_fallback

def enhance_satellite_image(img):
    """
    Cải thiện ảnh vệ tinh trong GIS
    """
    # Kiểm tra xem ảnh là màu hay xám
    is_color = len(img.shape) == 3
    
    if is_color:
        # Tách các kênh màu
        channels = cv2.split(img)
        enhanced_channels = []
        
        # Cải thiện từng kênh màu
        for channel in channels:
            # Cân bằng histogram với CLAHE
            enhanced_channel = clahe_equalization(channel, clip=2.0, grid=16)
            # Tăng cường độ sắc nét (giảm mức độ để tránh artifact)
            blurred = cv2.GaussianBlur(enhanced_channel, (5, 5), 0)
            enhanced_channel = cv2.addWeighted(enhanced_channel, 1.3, blurred, -0.3, 0)
            enhanced_channels.append(enhanced_channel)
        
        # Kết hợp các kênh màu đã xử lý
        enhanced = cv2.merge(enhanced_channels)
        
        # Tăng độ bão hòa màu một cách an toàn
        enhanced = enhanced.astype(np.float32)
        # Chuyển sang HSV và tăng saturation
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.15, 0, 255)  # Tăng 15% và clip
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    else:
        # Với ảnh xám, chỉ cần cân bằng histogram và tăng độ sắc nét
        enhanced = clahe_equalization(img, clip=2.0, grid=16)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        enhanced = cv2.addWeighted(enhanced, 1.3, blurred, -0.3, 0)
    
    return enhanced

def enhance_low_light_image(img):
    """
    Nâng cao chất lượng ảnh chụp trong điều kiện ánh sáng kém
    """
    # Kiểm tra xem ảnh là màu hay xám
    is_color = len(img.shape) == 3
    
    if is_color:
        # Chuyển sang không gian màu YCrCb để tách kênh độ sáng
        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        y_channel = ycrcb[:,:,0]
        
        # Áp dụng gamma correction cho kênh độ sáng
        y_enhanced = gamma_correction(y_channel, gamma=0.6)
        
        # Áp dụng CLAHE để cải thiện độ tương phản
        y_enhanced = clahe_equalization(y_enhanced, clip=3.0, grid=8)
        
        # Thay thế kênh độ sáng
        ycrcb[:,:,0] = y_enhanced
        
        # Chuyển lại sang RGB
        enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        
        # Giảm nhiễu đơn giản bằng Gaussian blur thay vì fastNlMeans
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    else:
        # Với ảnh xám, áp dụng gamma correction và CLAHE
        enhanced = gamma_correction(img, gamma=0.6)
        enhanced = clahe_equalization(enhanced, clip=3.0, grid=8)
        
        # Giảm nhiễu đơn giản
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    return enhanced

def restore_document_image(img, binary_output=False):
    """
    Khôi phục ảnh giấy tờ bị mờ, cũ, hoặc chất lượng kém
    Tập trung vào việc làm rõ văn bản mà không tạo quá nhiều noise
    
    Args:
        img: Ảnh đầu vào
        binary_output: True để trả về ảnh nhị phân, False để trả về ảnh xám cải thiện
    """
    try:
        # Bước 1: Chuyển sang ảnh xám
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()
        
        # Bước 2: Làm mượt nhẹ để giảm noise ban đầu
        if gray.shape[0] > 5 and gray.shape[1] > 5:
            smoothed = cv2.GaussianBlur(gray, (3, 3), 0)
        else:
            smoothed = gray.copy()
        
        # Bước 3: Hiệu chỉnh nền không đồng nhất (nhẹ nhàng hơn)
        # Tạo nền ước lượng bằng cách làm mờ vừa phải
        if smoothed.shape[0] > 35 and smoothed.shape[1] > 35:
            background = cv2.GaussianBlur(smoothed, (35, 35), 0)
        elif smoothed.shape[0] > 15 and smoothed.shape[1] > 15:
            background = cv2.GaussianBlur(smoothed, (15, 15), 0)
        elif smoothed.shape[0] > 5 and smoothed.shape[1] > 5:
            background = cv2.GaussianBlur(smoothed, (5, 5), 0)
        else:
            background = smoothed.copy()
        # Trừ nền một cách mềm mại
        normalized = cv2.absdiff(smoothed, background)
        # Chuẩn hóa
        normalized = cv2.normalize(normalized, None, 30, 220, cv2.NORM_MINMAX)
        
        # Bước 4: Tăng cường độ tương phản vừa phải
        enhanced = clahe_equalization(normalized, clip=2.5, grid=8)
        
        # Bước 5: Khử nhiễu thông minh nhưng giữ chi tiết
        # Bilateral filter với tham số conservative
        denoised = cv2.bilateralFilter(enhanced, 7, 40, 40)
        
        # Bước 6: Sắc nét hóa nhẹ (không quá mạnh)
        # Unsharp masking với mức độ vừa phải
        blurred = cv2.GaussianBlur(denoised, (2, 2), 1.0)
        sharpened = cv2.addWeighted(denoised, 1.3, blurred, -0.3, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        # Bước 7: Trả về kết quả theo yêu cầu
        if binary_output:
            # Trả về ảnh nhị phân với tham số conservative
            binary = cv2.adaptiveThreshold(sharpened, 255, 
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 
                                           21, 10)
            return binary
        else:
            # Trả về ảnh xám đã cải thiện (dễ đọc hơn)
            return sharpened
        
    except Exception as e:
        print(f"Lỗi trong restore_document_image: {e}")
        # Fallback: trả về ảnh xám với CLAHE nhẹ
        if len(img.shape) == 3:
            gray_fallback = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray_fallback = img.copy()
        
        enhanced_fallback = clahe_equalization(gray_fallback, clip=2.0, grid=8)
        
        if binary_output:
            binary_fallback = cv2.adaptiveThreshold(enhanced_fallback, 255, 
                                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                    cv2.THRESH_BINARY, 15, 5)
            return binary_fallback
        else:
            return enhanced_fallback
