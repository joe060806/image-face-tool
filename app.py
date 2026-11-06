import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="AI 輔助影像處理與人臉分析工具", layout="wide")
plt.rcParams["axes.unicode_minus"] = False

BG, PANEL, TXT, ACCENT, ACCENT2 = "#0b1020", "#141b34", "#e6eaf5", "#22d3ee", "#a78bfa"

st.markdown(f"""
<style>
.stApp {{ background: radial-gradient(1200px 600px at 10% -10%, #1b2550 0%, {BG} 55%); color:{TXT}; }}
[data-testid="stSidebar"] {{ background:{PANEL}; border-right:1px solid #26304f; }}
[data-testid="stHeader"] {{ background:transparent; }}
h1,h2,h3,h4,label,p,span,div {{ color:{TXT}; }}
.hero {{ padding:22px 28px; border-radius:18px; margin-bottom:14px;
  background:linear-gradient(120deg,{ACCENT}22,{ACCENT2}22); border:1px solid {ACCENT}55;
  box-shadow:0 0 30px {ACCENT}22; }}
.hero h1 {{ margin:0; font-size:2rem; background:linear-gradient(90deg,{ACCENT},{ACCENT2});
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
.hero p {{ margin:4px 0 0; opacity:.7; }}
.stTabs [data-baseweb="tab-list"] {{ gap:8px; }}
.stTabs [data-baseweb="tab"] {{ background:{PANEL}; border-radius:999px; padding:8px 20px;
  border:1px solid #26304f; }}
.stTabs [aria-selected="true"] {{ background:linear-gradient(90deg,{ACCENT}33,{ACCENT2}33);
  border-color:{ACCENT}; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display:none; }}
[data-testid="stMetric"] {{ background:{PANEL}; border:1px solid #26304f; border-radius:14px;
  padding:14px 16px; box-shadow:0 0 18px {ACCENT}11; }}
[data-testid="stMetricValue"] {{ color:{ACCENT}; }}
[data-testid="stImage"] img {{ border-radius:14px; border:1px solid #26304f; }}
[data-baseweb="popover"] [role="listbox"], [data-baseweb="popover"] ul {{ background:#f4f6fb !important; }}
[data-baseweb="popover"] li, [data-baseweb="popover"] li *, [data-baseweb="popover"] [role="option"] * {{
  color:#0b1020 !important; font-weight:600; }}
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] [aria-selected="true"] {{
  background:#dbeafe !important; }}
textarea {{ font-family:ui-monospace,Menlo,Consolas,monospace !important;
  background:{PANEL} !important; color:{ACCENT} !important; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- 工具函式
def load_image(file):
    raw = file.getvalue()
    img = Image.open(io.BytesIO(raw))
    info = {
        "檔名": file.name,
        "格式": img.format,
        "檔案大小": f"{len(raw) / 1024:.1f} KB",
        "解析度 (W×H)": f"{img.width} × {img.height}",
        "原始色彩模式": img.mode,
        "色彩維度 (通道數)": len(img.getbands()),
    }
    rgb = np.array(img.convert("RGB"))
    return rgb, info


def histogram_fig(data, labels, colors):
    """data: list of 2D uint8 arrays；橫軸 0-255，縱軸像素數量。"""
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for d, lb, c in zip(data, labels, colors):
        hist = cv2.calcHist([d], [0], None, [256], [0, 256]).ravel()
        ax.plot(np.arange(256), hist, color=c, label=lb)
        ax.fill_between(np.arange(256), hist, color=c, alpha=0.25)
    fig.patch.set_facecolor(PANEL)
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color("#26304f")
    ax.tick_params(colors=TXT)
    ax.set_xlim(0, 255)
    ax.set_xlabel("Pixel intensity (0-255)", color=TXT)
    ax.set_ylabel("Pixel count", color=TXT)
    ax.legend(facecolor=PANEL, edgecolor="#26304f", labelcolor=TXT)
    ax.grid(alpha=0.15, color="white")
    fig.tight_layout()
    return fig


def psnr(ref, test):
    """PSNR = 10*log10(255^2 / MSE)，MSE 為所有像素與通道的均方誤差。"""
    mse = np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


def parse_kernel(text):
    rows = [r for r in text.strip().splitlines() if r.strip()]
    k = np.array([[float(x) for x in r.replace(",", " ").split()] for r in rows])
    if k.ndim != 2 or k.shape[0] != k.shape[1] or k.shape[0] % 2 == 0:
        raise ValueError("卷積核必須為奇數階方陣（3×3、5×5、7×7…）")
    return k


PRESETS = {
    "Identity（恆等）": "0 0 0\n0 1 0\n0 0 0",
    "Sharpen（銳化）": "0 -1 0\n-1 5 -1\n0 -1 0",
    "Edge（邊緣偵測）": "-1 -1 -1\n-1 8 -1\n-1 -1 -1",
    "Emboss（浮雕）": "-2 -1 0\n-1 1 1\n0 1 2",
    "Box Blur 3×3": "1 1 1\n1 1 1\n1 1 1",
    "Gaussian 3×3": "1 2 1\n2 4 2\n1 2 1",
    "Gaussian 5×5": "1 4 6 4 1\n4 16 24 16 4\n6 24 36 24 6\n4 16 24 16 4\n1 4 6 4 1",
    "Sobel X": "-1 0 1\n-2 0 2\n-1 0 1",
}


def convolve(rgb, kernel, normalize, bias):
    k = kernel.copy()
    if normalize:
        s = k.sum()
        if abs(s) > 1e-8:
            k = k / s
    # cv2.filter2D 實為相關(correlation)，翻轉核後即為真正卷積
    k_flip = cv2.flip(k, -1)
    out = cv2.filter2D(rgb.astype(np.float32), -1, k_flip,
                       borderType=cv2.BORDER_REFLECT_101) + bias
    return np.clip(out, 0, 255).astype(np.uint8), k


@st.cache_resource
def haar_model():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_haar(rgb, scale, neighbors, min_size):
    gray = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    res = haar_model().detectMultiScale3(
        gray, scaleFactor=scale, minNeighbors=neighbors,
        minSize=(min_size, min_size), outputRejectLevels=True)
    boxes, _, weights = res
    out = []
    for (x, y, w, h), lw in zip(boxes, np.ravel(weights)):
        conf = 1 / (1 + np.exp(-(float(lw) - 3) / 1.5))  # levelWeight 映射為 0~1 的相對信心度
        out.append((int(x), int(y), int(w), int(h), float(conf)))
    return out


def detect_mediapipe(rgb, min_conf):
    import mediapipe as mp
    H, W = rgb.shape[:2]
    out = []
    with mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=min_conf) as fd:
        r = fd.process(rgb)
        for d in (r.detections or []):
            b = d.location_data.relative_bounding_box
            x, y = max(int(b.xmin * W), 0), max(int(b.ymin * H), 0)
            out.append((x, y, int(b.width * W), int(b.height * H), float(d.score[0])))
    return out


# ---------------------------------------------------------------- 介面
st.markdown('<div class="hero"><h1>🖼️ AI 輔助之互動式影像處理與人臉分析工具</h1>'
            '<p>Color Space · Histogram · Convolution · PSNR · Face Detection</p></div>',
            unsafe_allow_html=True)
file = st.sidebar.file_uploader("上傳影像 (PNG / JPG / JPEG / BMP)",
                                type=["png", "jpg", "jpeg", "bmp"])
if file is None:
    st.info("請先於左側上傳影像。")
    st.stop()

rgb, info = load_image(file)
st.sidebar.subheader("影像中繼資訊")
for k, v in info.items():
    st.sidebar.write(f"**{k}**：{v}")

tab1, tab2, tab3 = st.tabs(["① 色彩空間與直方圖", "② 卷積濾波與 PSNR", "③ 人臉偵測"])

# ---- Tab 1
with tab1:
    mode = st.radio("檢視模式",
                    ["Original", "R", "G", "B", "Y", "U", "V"], horizontal=True)
    yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
    c1, c2 = st.columns(2)
    if mode == "Original":
        show = rgb
        hist_fig = histogram_fig([rgb[..., 0], rgb[..., 1], rgb[..., 2]],
                                 ["R", "G", "B"], ["#ff5c7a", "#4ade80", "#38bdf8"])
        caption = "原始影像（直方圖為 R/G/B 疊合）"
    elif mode in "RGB":
        i = "RGB".index(mode)
        show = rgb[..., i]
        hist_fig = histogram_fig([show], [mode], [["#ff5c7a", "#4ade80", "#38bdf8"][i]])
        caption = f"{mode} 通道（灰階強度）"
    else:
        i = "YUV".index(mode)
        show = yuv[..., i]
        hist_fig = histogram_fig([show], [mode], ["#e6eaf5", "#38bdf8", "#fb923c"][i:i + 1])
        caption = f"{mode} 分量（YUV；U/V 以 128 為零點偏移）"
    c1.image(show, caption=caption, use_container_width=True, clamp=True)
    c2.pyplot(hist_fig)

# ---- Tab 2
with tab2:
    left, right = st.columns([1, 2])
    with left:
        preset = st.selectbox("預設卷積核（可再手動修改）", list(PRESETS))
        text = st.text_area("卷積核矩陣（每列一行，數值以空白或逗號分隔）",
                            PRESETS[preset], height=170, key=f"k_{preset}")
        normalize = st.checkbox("正規化（除以核權重總和；總和為 0 時不處理）", True)
        bias = st.slider("亮度偏移 (bias)", -128, 128, 0)
    try:
        kernel = parse_kernel(text)
        filtered, used = convolve(rgb, kernel, normalize, bias)
        left.caption(f"實際使用的核（{kernel.shape[0]}×{kernel.shape[0]}）")
        left.dataframe(used.round(4))
        with right:
            view = st.radio("比對方式", ["並排", "差異圖 |原圖−濾波後|"], horizontal=True)
            if view == "並排":
                a, b = st.columns(2)
                a.image(rgb, caption="濾波前", use_container_width=True)
                b.image(filtered, caption="濾波後", use_container_width=True)
            else:
                diff = cv2.absdiff(rgb, filtered)
                st.image(np.clip(diff.astype(int) * 4, 0, 255).astype(np.uint8),
                         caption="差異圖（放大 4 倍以利觀察）", use_container_width=True)
            p = psnr(rgb, filtered)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("PSNR (整體)", "∞ dB" if np.isinf(p) else f"{p:.2f} dB")
            for col, n, i in zip((m2, m3, m4), "RGB", range(3)):
                pc = psnr(rgb[..., i], filtered[..., i])
                col.metric(f"PSNR ({n})", "∞ dB" if np.isinf(pc) else f"{pc:.2f} dB")
            st.latex(r"PSNR = 10\log_{10}\frac{255^2}{MSE},\quad "
                     r"MSE=\frac{1}{HWC}\sum (I-K)^2")
            fig = histogram_fig([cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
                                 cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)],
                                ["Before", "After"], ["#94a3b8", "#22d3ee"])
            st.pyplot(fig)
    except Exception as e:
        right.error(f"卷積核格式錯誤：{e}")

# ---- Tab 3
with tab3:
    algo = st.radio("偵測演算法", ["Haar Cascade (OpenCV)", "MediaPipe Face Detection"],
                    horizontal=True)
    if algo.startswith("Haar"):
        a, b, c = st.columns(3)
        scale = a.slider("scaleFactor", 1.05, 1.5, 1.1, 0.05)
        nb = b.slider("minNeighbors", 1, 10, 5)
        ms = c.slider("最小人臉尺寸 (px)", 20, 200, 40)
        faces = detect_haar(rgb, scale, nb, ms)
    else:
        mc = st.slider("最小信心度", 0.1, 0.9, 0.5, 0.05)
        try:
            faces = detect_mediapipe(rgb, mc)
        except Exception as e:
            st.error(f"MediaPipe 無法使用（{e}），請 pip install mediapipe 或改用 Haar。")
            faces = []
    canvas = rgb.copy()
    th = max(2, rgb.shape[1] // 300)
    for i, (x, y, w, h, cf) in enumerate(faces, 1):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), th)
        cv2.putText(canvas, f"#{i} {cf:.2f}", (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.6, rgb.shape[1] / 1200),
                    (255, 0, 0), th)
    st.metric("偵測到的人臉數量", len(faces))
    st.image(canvas, use_container_width=True)
    if faces:
        st.dataframe([{"編號": i, "x": f[0], "y": f[1], "寬": f[2], "高": f[3],
                       "信心度": round(f[4], 3)} for i, f in enumerate(faces, 1)])
    if algo.startswith("Haar"):
        st.caption("註：Haar 本身不輸出機率，信心度由 cascade 的 levelWeights 經 sigmoid 映射而得，為相對指標。")