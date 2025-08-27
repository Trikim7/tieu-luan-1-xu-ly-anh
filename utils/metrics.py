import numpy as np

def _to_gray_if_needed(arr: np.ndarray) -> np.ndarray:
    """
    Convert to grayscale if input is RGB-like. Assumes uint8 [0,255].
    """
    if arr.ndim == 3 and arr.shape[2] >= 3:
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        return y.astype(np.uint8)
    return arr

def _ensure_same_shape(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Convert to grayscale when channel mismatch
    if a.ndim != b.ndim:
        a = _to_gray_if_needed(a)
        b = _to_gray_if_needed(b)
    # If shapes still differ, crop to min common area
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a = a[:h, :w]
    b = b[:h, :w]
    return a, b

def compute_mse(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _ensure_same_shape(a, b)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    diff = a - b
    return float(np.mean(diff * diff))

def compute_psnr(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    mse = compute_mse(a, b)
    if mse == 0:
        return float('inf')
    return 20.0 * float(np.log10(max_val)) - 10.0 * float(np.log10(mse))
