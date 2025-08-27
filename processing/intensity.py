import numpy as np

def negative(img):
    return 255 - img

def log_transform(img):
    """
    Biến đổi logarithm để tăng cường vùng tối
    s = c * log(1 + r)
    """
    # Đảm bảo ảnh đầu vào là float để tính toán chính xác
    img_float = img.astype(np.float64)
    
    # Tính hằng số c để đảm bảo giá trị output trong khoảng [0, 255]
    max_val = np.max(img_float)
    if max_val == 0:
        # Ảnh toàn đen, trả về ảnh gốc
        return img
    
    c = 255.0 / np.log(1.0 + max_val)
    
    # Áp dụng biến đổi log
    log_img = c * np.log(1.0 + img_float)
    
    # Đảm bảo giá trị trong khoảng [0, 255] và chuyển về uint8
    log_img = np.clip(log_img, 0, 255)
    
    return log_img.astype(np.uint8)

def gamma_correction(img, gamma=1.0):
    return np.array(255 * (img / 255) ** gamma, dtype=np.uint8)

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
