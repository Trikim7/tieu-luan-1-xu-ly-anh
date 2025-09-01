import numpy as np
from PIL import Image

def hist_equalization(img):
    """
    Cân bằng lược đồ mức xám toàn cục (Global Histogram Equalization)
    """
    # Kiểm tra đầu vào phải là ảnh xám kiểu uint8
    if len(img.shape) != 2 or img.dtype != np.uint8:
        raise ValueError("Đầu vào phải là ảnh xám (grayscale) với kiểu dữ liệu uint8")
    # Tính histogram của ảnh
    hist, bins = np.histogram(img.flatten(), 256, [0,256])
    # Tính hàm phân phối tích lũy (CDF)
    cdf = hist.cumsum()
    # Loại bỏ các giá trị bằng 0 trong CDF
    cdf_masked = np.ma.masked_equal(cdf, 0)
    # Tìm giá trị nhỏ nhất và lớn nhất của CDF
    cdf_min = cdf_masked.min()
    cdf_max = cdf_masked.max()
    # Chuẩn hóa CDF về khoảng [0, 255]
    cdf_masked = (cdf_masked - cdf_min) * 255 / (cdf_max - cdf_min)
    # Điền lại các giá trị đã loại bỏ bằng 0 và chuyển về kiểu uint8
    cdf_final = np.ma.filled(cdf_masked, 0).astype('uint8')
    # Tra cứu giá trị mới cho từng pixel dựa vào CDF
    img_eq = cdf_final[img]
    return img_eq

def clahe_equalization(img, clip=2.0, grid=8):
    """
    Cân bằng lược đồ mức xám thích ứng có giới hạn (CLAHE - Contrast Limited Adaptive Histogram Equalization)
    
    Args:
        img: Ảnh xám đầu vào (uint8)
        clip: Giới hạn clipping cho histogram
        grid: Số lượng tile theo mỗi chiều (grid x grid)
    """
    # Kiểm tra đầu vào phải là ảnh xám kiểu uint8
    if len(img.shape) != 2 or img.dtype != np.uint8:
        raise ValueError("Đầu vào phải là ảnh xám (grayscale) với kiểu dữ liệu uint8")
    h, w = img.shape
    # Chia ảnh thành các vùng nhỏ (tile)
    tile_h, tile_w = h // grid, w // grid
    result = np.zeros_like(img)
    for i in range(grid):
        for j in range(grid):
            # Xác định vùng tile hiện tại
            y0, y1 = i * tile_h, (i + 1) * tile_h if i < grid - 1 else h
            x0, x1 = j * tile_w, (j + 1) * tile_w if j < grid - 1 else w
            tile = img[y0:y1, x0:x1]
            # Tính histogram cho tile
            hist, bins = np.histogram(tile.flatten(), 256, [0,256])
            # Giới hạn giá trị histogram (clip)
            clip_limit = int(clip * tile.size / 256)
            excess = hist - clip_limit  # Tính phần dư vượt quá clip
            excess[excess < 0] = 0
            n_excess = excess.sum()  # Tổng phần dư
            hist = np.minimum(hist, clip_limit)  # Áp dụng clip
            # Phân phối lại phần dư cho các mức xám
            hist += n_excess // 256
            # Tính CDF cho tile
            cdf = hist.cumsum()
            cdf_masked = np.ma.masked_equal(cdf, 0)
            cdf_min = cdf_masked.min()
            cdf_max = cdf_masked.max()
            cdf_masked = (cdf_masked - cdf_min) * 255 / (cdf_max - cdf_min)
            cdf_final = np.ma.filled(cdf_masked, 0).astype('uint8')
            # Tra cứu giá trị mới cho tile dựa vào CDF
            result[y0:y1, x0:x1] = cdf_final[tile]
    # Trả về ảnh sau CLAHE
    return result

def ahe_equalization(img, window_size=64):
    """
    Cân bằng lược đồ mức xám thích ứng (Adaptive Histogram Equalization - AHE)
    Tối ưu hóa cho tốc độ bằng cách sử dụng vectorization
    
    Args:
        img: Ảnh xám đầu vào (uint8)
        window_size: Kích thước cửa sổ local (mặc định 64x64)
    """
    if len(img.shape) != 2 or img.dtype != np.uint8:
        raise ValueError("Đầu vào phải là ảnh xám (grayscale) với kiểu dữ liệu uint8")
    
    h, w = img.shape
    result = np.zeros_like(img, dtype=np.uint8)
    
    # Pad ảnh để xử lý biên
    pad_size = window_size // 2
    padded_img = np.pad(img, pad_size, mode='reflect')
    
    # Tối ưu: Xử lý theo batch để giảm overhead
    batch_size = min(50, h)  # Xử lý 50 hàng một lúc
    
    for batch_start in range(0, h, batch_size):
        batch_end = min(batch_start + batch_size, h)
        
        for i in range(batch_start, batch_end):
            for j in range(0, w, 10):  # Skip một số pixel để tăng tốc
                j_end = min(j + 10, w)
                
                # Lấy window local quanh pixel (i,j)
                y_start = i
                y_end = i + window_size
                x_start = j  
                x_end = j + window_size
                
                local_window = padded_img[y_start:y_end, x_start:x_end]
                
                # Tính histogram cho window local
                hist, _ = np.histogram(local_window.flatten(), 256, [0, 256])
                
                # Tính CDF
                cdf = hist.cumsum()
                
                # Loại bỏ giá trị 0 và chuẩn hóa
                cdf_masked = np.ma.masked_equal(cdf, 0)
                if cdf_masked.min() == cdf_masked.max():
                    # Trường hợp đặc biệt: tất cả pixel có cùng giá trị
                    result[i, j:j_end] = img[i, j:j_end]
                else:
                    cdf_min = cdf_masked.min()
                    cdf_max = cdf_masked.max()
                    cdf_normalized = (cdf_masked - cdf_min) * 255 / (cdf_max - cdf_min)
                    cdf_final = np.ma.filled(cdf_normalized, 0).astype('uint8')
                    
                    # Áp dụng transformation cho batch pixel
                    for jj in range(j, j_end):
                        result[i, jj] = cdf_final[img[i, jj]]
    
    return result

def auto_optimize_ahe_params(img):
    """
    Tự động tối ưu parameters cho AHE dựa trên đặc điểm ảnh
    """
    h, w = img.shape
    
    # 1. Tính window size dựa trên kích thước ảnh
    # Ảnh nhỏ -> window nhỏ, ảnh lớn -> window lớn
    img_size = h * w
    if img_size < 100000:  # < 100K pixels
        window_size = 32
        step_size = 4
    elif img_size < 500000:  # < 500K pixels  
        window_size = 64
        step_size = 6
    elif img_size < 1000000:  # < 1M pixels
        window_size = 96
        step_size = 8
    else:  # >= 1M pixels
        window_size = 128
        step_size = 12
    
    # 2. Tính contrast dựa trên histogram
    hist = np.histogram(img, bins=256, range=(0, 255))[0]
    hist_norm = hist / hist.sum()
    
    # Tính entropy (measure of uniformity)
    entropy = -np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0]))
    
    # 3. Adjust parameters dựa trên entropy
    # Low entropy (ít contrast) -> cần window nhỏ hơn để enhance local details
    # High entropy (nhiều contrast) -> window lớn hơn để tránh over-enhancement
    if entropy < 6:  # Low contrast
        window_size = max(32, window_size // 2)
        step_size = max(4, step_size // 2)
    elif entropy > 7.5:  # High contrast
        window_size = min(128, int(window_size * 1.2))
        step_size = min(16, int(step_size * 1.5))
    
    # 4. Kiểm tra noise level (dựa trên local variance)
    # Sample một số vùng để tính variance
    sample_size = 32
    variances = []
    for i in range(5):  # Sample 5 vùng
        start_row = np.random.randint(0, max(1, h - sample_size))
        start_col = np.random.randint(0, max(1, w - sample_size))
        sample = img[start_row:start_row+sample_size, start_col:start_col+sample_size]
        variances.append(np.var(sample))
    
    avg_variance = np.mean(variances)
    
    # Nếu nhiều noise -> tăng step_size để smooth hơn
    if avg_variance > 1000:  # High noise
        step_size = min(16, int(step_size * 1.3))
    
    return window_size, step_size
    """
    Phiên bản nhanh của AHE - tối ưu hóa để tránh treo web
    """
    if len(img.shape) != 2 or img.dtype != np.uint8:
        raise ValueError("Đầu vào phải là ảnh xám (grayscale) với kiểu dữ liệu uint8")
    
    h, w = img.shape
    
    # Giới hạn kích thước để tránh quá chậm
    if h * w > 1000000:  # > 1M pixels
        scale = np.sqrt(1000000 / (h * w))
        new_h, new_w = int(h * scale), int(w * scale)
        img = np.array(Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS))
        h, w = new_h, new_w
    
    result = np.zeros_like(img, dtype=np.uint8)
    
    # Tăng step_size để xử lý nhanh hơn
    step_size = max(step_size, min(h, w) // 20)
    window_size = min(window_size, min(h, w) // 4)
    
    # Pad ảnh
    pad_size = window_size // 2
    padded_img = np.pad(img, pad_size, mode='reflect')
    
    # Tạo grid và transformations
    grid_points = []
    transformations = {}
    
    for i in range(0, h, step_size):
        for j in range(0, w, step_size):
            y_start, y_end = i, i + window_size
            x_start, x_end = j, j + window_size
            
            local_window = padded_img[y_start:y_end, x_start:x_end]
            
            # Tính histogram nhanh
            hist = np.bincount(local_window.ravel(), minlength=256)
            cdf = np.cumsum(hist)
            
            # Skip nếu không có variation
            if cdf[-1] == cdf[0]:
                transformations[(i, j)] = np.arange(256, dtype=np.uint8)
                continue
            
            # Normalize CDF nhanh
            cdf_norm = ((cdf - cdf[cdf > 0].min()) * 255 / 
                       (cdf[-1] - cdf[cdf > 0].min())).astype(np.uint8)
            cdf_norm[cdf == 0] = 0
            
            transformations[(i, j)] = cdf_norm
            grid_points.append((i, j))
    
    # Áp dụng transformation bằng nearest neighbor nhanh
    for i in range(h):
        for j in range(w):
            # Tìm grid point gần nhất nhanh
            closest_i = min(grid_points, key=lambda p: abs(p[0] - i))[0]
            closest_j = min([p for p in grid_points if p[0] == closest_i], 
                           key=lambda p: abs(p[1] - j))[1]
            
            if (closest_i, closest_j) in transformations:
                result[i, j] = transformations[(closest_i, closest_j)][img[i, j]]
            else:
                result[i, j] = img[i, j]
    
    return result

def ahe_equalization_fast(img, window_size=None, step_size=None):
    """
    AHE tối ưu tốc độ với auto parameters
    """
    if window_size is None or step_size is None:
        # Tự động tối ưu parameters
        auto_window, auto_step = auto_optimize_ahe_params(img)
        window_size = window_size or auto_window
        step_size = step_size or auto_step
    
    if len(img.shape) != 2 or img.dtype != np.uint8:
        raise ValueError("Đầu vào phải là ảnh xám (grayscale) với kiểu dữ liệu uint8")
    
    h, w = img.shape
    result = img.copy().astype(np.float32)
    
    # Tối ưu: Giới hạn kích thước ảnh để tránh quá chậm
    if h * w > 1000000:  # > 1M pixels
        # Resize xuống để xử lý nhanh
        from PIL import Image
        scale = np.sqrt(1000000 / (h * w))
        new_h, new_w = int(h * scale), int(w * scale)
        img_small = np.array(Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS))
        
        # Xử lý ảnh nhỏ
        result_small = ahe_equalization_fast(img_small, 
                                           max(32, window_size // 2), 
                                           max(4, step_size // 2))
        
        # Resize lại về kích thước gốc
        result = np.array(Image.fromarray(result_small).resize((w, h), Image.LANCZOS))
        return result.astype(np.uint8)
    
    # AHE processing với step_size để tăng tốc
    half_window = window_size // 2
    
    for i in range(0, h, step_size):
        for j in range(0, w, step_size):
            # Xác định vùng local
            y1 = max(0, i - half_window)
            y2 = min(h, i + half_window)
            x1 = max(0, j - half_window)
            x2 = min(w, j + half_window)
            
            # Lấy local region
            local_region = img[y1:y2, x1:x2]
            
            # Tính histogram và CDF
            hist = np.histogram(local_region, bins=256, range=(0, 255))[0]
            cdf = np.cumsum(hist).astype(np.float32)
            
            # Normalize CDF
            cdf_min = cdf[cdf > 0].min() if (cdf > 0).any() else 0
            cdf_normalized = (cdf - cdf_min) * 255 / (cdf[-1] - cdf_min) if cdf[-1] > cdf_min else cdf
            
            # Áp dụng cho vùng step_size x step_size
            i_end = min(i + step_size, h)
            j_end = min(j + step_size, w)
            
            for ii in range(i, i_end):
                for jj in range(j, j_end):
                    if 0 <= ii < h and 0 <= jj < w:
                        pixel_val = img[ii, jj]
                        result[ii, jj] = cdf_normalized[pixel_val]
    
    return np.clip(result, 0, 255).astype(np.uint8)
