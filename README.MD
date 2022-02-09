# JPEG Decoder

本篇為2020中央資料壓縮(Data Compression)作業之一，當初覺得使用雙迴圈進行DCT太慢，排除python本身效能不彰的問題，開始尋找演算法上的改進，直到現在才有空補完從RGB圖檔輸出成JPG的過程。

## 使用方法
```python
wget https://pbs.twimg.com/profile_images/1482580721466486784/q36x-Uux_400x400.jpg -o 400x400.jpg
python3 decoder.py
```

## 開發紀錄
- Part 1. https://hackmd.io/@93i7xo2/jpeg-decoder-1
- Part 2. https://hackmd.io/@93i7xo2/jpeg-decoder-2
- Part 3. https://hackmd.io/@93i7xo2/jpeg-decoder-3