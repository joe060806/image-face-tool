# AI 輔助之互動式影像處理與人臉分析工具

## 安裝與執行
```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
（MediaPipe 為選用；若安裝失敗，可從 requirements 移除，程式仍可用 Haar 偵測。）

## 功能對照
| 需求 | 實作 |
|---|---|
| 1. 影像載入 | PNG/JPG/JPEG/BMP；顯示檔名、格式、檔案大小、解析度、色彩模式與通道數 |
| 2. 色彩空間 | Original / R / G / B / Y / U / V 切換，右側同步繪製 0–255 直方圖 |
| 3. 卷積濾波 | 自訂 3×3、5×5…奇數階核、預設範本、正規化、bias；並排/差異圖；PSNR |
| 4. 人臉偵測 | Haar Cascade（OpenCV）或 MediaPipe；邊界框、人臉總數、信心度 |

## 公式

**RGB → YUV（OpenCV, BT.601, 8-bit）**

Y = 0.299R + 0.587G + 0.114B
U = −0.147R − 0.289G + 0.436B + 128 (以 0.492(B−Y)+128 等價表示)
V = 0.615R − 0.515G − 0.100B + 128 (以 0.877(R−Y)+128 等價表示)

**二維離散卷積**

(I * K)(x,y) = Σᵢ Σⱼ I(x−i, y−j)·K(i,j)

實作以 `cv2.filter2D`（相關運算）搭配核翻轉 180° 得到真正卷積；邊界採 reflect。
正規化：K' = K / ΣK（ΣK = 0，如邊緣偵測核，則不正規化）。

**PSNR**

MSE = (1 / (H·W·C)) · Σ (I(x,y,c) − K(x,y,c))²

PSNR = 10 · log₁₀( MAX² / MSE ) = 20 · log₁₀( MAX / √MSE )，MAX = 255（8-bit），單位 dB。
MSE = 0（兩圖完全相同，如恆等核）時 PSNR → ∞。濾波結果先 clip 到 [0,255] 並轉 uint8 後再計算。

## 信心度說明
- MediaPipe：直接使用模型輸出之 detection score（0–1）。
- Haar：cascade 不提供機率，使用 `detectMultiScale3` 的 levelWeights，經 sigmoid 映射為 0–1 相對指標。

## 與 AI 協作紀錄（請依實際情況補充於報告）
1. 需求拆解與架構設計（Streamlit 三分頁）
2. 各模組程式生成與參數調校
3. 測試案例：恆等核（PSNR=∞）、高斯平滑（PSNR 高）、銳化/邊緣（PSNR 低）
4. 問題與修正：如 filter2D 為相關而非卷積、YUV 的 U/V 偏移顯示等
