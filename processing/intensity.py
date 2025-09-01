import numpy as np
import cv2

def negative(img):
    return 255 - img

def log_transform(img, c=1):
    """
    Thực hiện biến đổi logarithm: s = c * log(1 + r)
    
    Args:
        img: Ảnh đầu vào (grayscale, uint8)
        c: Hệ số scaling (range từ 0.1 đến 50 để thấy rõ sự khác biệt)
    """
    # Chuyển ảnh về float64 để tránh overflow với c lớn
    img_float = img.astype(np.float64) / 255.0
    
    # Áp dụng log transform với range rộng hơn
    if c > 10:
        # Với c lớn, sử dụng scaling đặc biệt để tránh overflow
        log_img = c * np.log1p(img_float * 10) / 10
    else:
        # Với c nhỏ, dùng công thức thông thường
        log_img = c * np.log1p(img_float)
    
    # Chuẩn hóa kết quả về [0, 255] để tận dụng toàn bộ dynamic range
    log_img = cv2.normalize(log_img, None, 0, 255, cv2.NORM_MINMAX)
    
    # Chuyển về uint8 cho hiển thị
    return log_img.astype(np.uint8)

def gamma_correction(img, gamma=1.0, c=1.0):
    """
    Gamma correction (Power-law transformation): s = c * r^gamma
    
    Args:
        img: Ảnh đầu vào
        gamma: Tham số power
            - gamma < 1: Tăng cường vùng tối (dark regions)
            - gamma > 1: Tăng cường vùng sáng (bright regions) 
            - gamma = 1: Không thay đổi (linear)
        c: Hằng số scaling (mặc định = 1)
    """
    # Chuẩn hóa về [0,1]
    img_normalized = img.astype(np.float32) / 255.0
    
    # Áp dụng power-law: s = c * r^gamma
    transformed = c * np.power(img_normalized, gamma)
    
    # Clip và chuyển về [0,255]
    transformed = np.clip(transformed * 255, 0, 255)
    
    return transformed.astype(np.uint8)

def _piecewise_lut(r1: int, s1: int, r2: int, s2: int) -> np.ndarray:
    """
    Tạo bảng tra cứu (LUT) 256 mức cho biến đổi tuyến tính từng đoạn.
    Ba đoạn: [0, r1], [r1, r2], [r2, 255].
    """
    r1 = int(np.clip(r1, 0, 255))
    r2 = int(np.clip(r2, 0, 255))
    s1 = int(np.clip(s1, 0, 255))
    s2 = int(np.clip(s2, 0, 255))
    if r2 <= r1:
        # Tránh chia 0, đảm bảo r2 > r1
        r2 = min(255, r1 + 1)

    lut = np.zeros(256, dtype=np.float32)
    # Đoạn 1
    if r1 > 0:
        lut[:r1+1] = np.linspace(0, s1, r1+1)
    else:
        lut[0] = s1
    # Đoạn 2
    mid_len = max(1, r2 - r1)
    lut[r1:r2+1] = np.linspace(s1, s2, mid_len + 1)
    # Đoạn 3
    if r2 < 255:
        lut[r2:256] = np.linspace(s2, 255, 256 - r2)
    else:
        lut[255] = s2
    return np.clip(lut, 0, 255).astype(np.uint8)

def piecewise_linear(img: np.ndarray, r1: int, s1: int, r2: int, s2: int) -> np.ndarray:
    """
    Biến đổi tuyến tính từng đoạn (Piecewise-linear).
    - Áp dụng LUT cho ảnh xám hoặc từng kênh của ảnh màu.
    - Thường dùng cho contrast stretching.
    """
    lut = _piecewise_lut(r1, s1, r2, s2)
    if img.ndim == 2:
        return lut[img]
    elif img.ndim == 3:
        # Áp dụng theo từng kênh
        channels = [lut[img[..., c]] for c in range(img.shape[2])]
        return np.stack(channels, axis=-1)
    else:
        raise ValueError("Ảnh đầu vào phải là ảnh xám hoặc ảnh màu RGB")
