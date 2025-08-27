import numpy as np

def hist_equalization(img):
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
