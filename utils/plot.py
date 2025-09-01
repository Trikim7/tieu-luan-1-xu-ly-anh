import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
from PIL import Image
import hashlib

# Cache cho histogram để tránh tính toán lại
_histogram_cache = {}

def get_image_hash(img):
    """Tạo hash của ảnh để cache"""
    return hashlib.md5(img.tobytes()).hexdigest()[:16]

def plot_histogram_cached(img_hash, img_shape, hist_data):
    """Vẽ histogram với cache đơn giản"""
    try:
        fig, ax = plt.subplots(figsize=(5, 2.5))
        
        bin_centers, hist = hist_data
        ax.bar(bin_centers, hist, width=2, color='black', alpha=0.7)
        ax.set_title("Histogram", fontsize=10)
        ax.set_xlim(0, 255)
        
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=80, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        return Image.open(buf)
    except Exception:
        return Image.new('RGB', (400, 200), 'white')

def plot_histogram(img):
    """
    Vẽ histogram với caching và tối ưu hóa
    """
    try:
        # Chuyển sang grayscale nếu là ảnh màu
        if len(img.shape) == 3:
            img = np.dot(img[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
        
        # Tạo hash để cache
        img_hash = get_image_hash(img)
        
        # Kiểm tra cache
        if img_hash in _histogram_cache:
            return _histogram_cache[img_hash]
        
        # Downsample nếu ảnh quá lớn
        if img.size > 500000:
            step = int(np.sqrt(img.size / 100000))
            img = img[::step, ::step]
        
        # Tính histogram
        hist, bins = np.histogram(img.ravel(), bins=128, range=(0,256))
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Vẽ với cache
        result = plot_histogram_cached(img_hash, img.shape, (bin_centers, hist))
        
        # Lưu vào cache (giới hạn 10 items)
        if len(_histogram_cache) > 10:
            _histogram_cache.clear()
        _histogram_cache[img_hash] = result
        
        return result
    
    except Exception as e:
        print(f"Lỗi plot_histogram: {e}")
        return Image.new('RGB', (400, 200), 'white')

def plot_histogram_streamlit(img):
    """
    Vẽ histogram trực tiếp cho Streamlit
    Trả về matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Kiểm tra ảnh màu hay xám
    if len(img.shape) == 3:
        # Ảnh màu - vẽ histogram cho từng kênh
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            ax.hist(img[:,:,i].ravel(), bins=256, range=(0,256), 
                   color=color, alpha=0.7, label=f'Channel {color}')
        ax.legend()
    else:
        # Ảnh xám
        ax.hist(img.ravel(), bins=256, range=(0,256), color='black', alpha=0.7)
    
    ax.set_title("Histogram", fontsize=14)
    ax.set_xlabel("Pixel Intensity", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    return fig

def create_histogram_array(img):
    """
    Tạo histogram array để vẽ bằng st.bar_chart (phương pháp backup)
    """
    if len(img.shape) == 3:
        # Chỉ lấy kênh đầu tiên nếu là ảnh màu
        img = img[:,:,0]
    
    hist, bins = np.histogram(img.ravel(), bins=256, range=(0, 256))
    return hist

def plot_histogram_simple(img):
    """
    Tạo histogram đơn giản không dùng matplotlib (backup method)
    Trả về data cho st.bar_chart
    """
    if len(img.shape) == 3:
        # Convert to grayscale
        img_gray = np.dot(img[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
    else:
        img_gray = img
    
    hist, bins = np.histogram(img_gray.ravel(), bins=256, range=(0, 256))
    
    # Tạo DataFrame cho streamlit
    import pandas as pd
    df = pd.DataFrame({
        'Pixel Value': range(256),
        'Count': hist
    })
    return df

def show_histogram_safe(img, title="Histogram"):
    """
    Hiển thị histogram một cách an toàn với fallback
    """
    import streamlit as st
    
    try:
        # Thử dùng matplotlib trước
        fig = plot_histogram_streamlit(img)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Không thể vẽ histogram bằng matplotlib: {e}")
        try:
            # Fallback sang st.bar_chart
            df = plot_histogram_simple(img)
            st.bar_chart(df.set_index('Pixel Value')['Count'])
        except Exception as e2:
            st.error(f"Không thể vẽ histogram: {e2}")
            # Hiển thị thông tin cơ bản
            if len(img.shape) == 3:
                img_gray = np.dot(img[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
            else:
                img_gray = img
            st.write(f"**{title} - Thống kê cơ bản:**")
            st.write(f"- Min: {img_gray.min()}")
            st.write(f"- Max: {img_gray.max()}")
            st.write(f"- Mean: {img_gray.mean():.2f}")
            st.write(f"- Std: {img_gray.std():.2f}")
