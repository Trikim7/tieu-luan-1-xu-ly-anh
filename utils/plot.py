import matplotlib.pyplot as plt
from io import BytesIO

def plot_histogram(img):
    fig, ax = plt.subplots()
    ax.hist(img.ravel(), bins=256, range=(0,256), color='black')
    ax.set_title("Histogram")
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf
