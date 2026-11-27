"""
功能：
1. 空間局部性 (Spatial Locality) 分析：連續存取判定、尋道偏移量 (Offset Delta)
2. 時間局部性 (Temporal Locality) 分析：4KB 區塊劃分、重用率 (Reuse Ratio)
3. 磁碟排程模擬 (Disk Scheduling Simulation)：FCFS, SSTF, LOOK 演算法
"""

from typing import Dict, List, Literal, Tuple
import numpy as np
import pandas as pd

# ==============================================================================
# 模組一：Locality（局部性）判定與特徵分析
# ==============================================================================


def analyze_spatial_locality(
    df: pd.DataFrame, threshold_bytes: int = 65536
) -> Dict[str, float]:
  """計算空間局部性相關指標

  :param df: 包含 'Offset' 與 'Size' 欄位的 DataFrame (單位: Bytes)
  :param threshold_bytes: 判定為局部相鄰存取的距離上限 (預設 64 KB)
  """
  # 計算上一個請求的結束位址
  prev_end = (df["Offset"] + df["Size"]).shift(1)

  # 計算相鄰請求的偏移量距離 Delta
  offset_delta = df["Offset"] - prev_end

  # 1. 嚴格連續 (Strictly Sequential): 當前起始 == 前一個結束
  is_strict_seq = df["Offset"] == prev_end

  # 2. 半連續 / 局部存取 (Local Access): 偏移量在特定閾值內
  is_local = (offset_delta.abs() <= threshold_bytes) & (~is_strict_seq)

  # 3. 隨機存取 (Random Access): 偏移量超出閾值
  is_random = offset_delta.abs() > threshold_bytes

  return {
      "total_requests": len(df),
      "strict_sequential_ratio_%": float(is_strict_seq.mean() * 100),
      "local_adjacent_ratio_%": float(is_local.mean() * 100),
      "random_access_ratio_%": float(is_random.mean() * 100),
      "mean_absolute_seek_delta_mb": float(
          offset_delta.abs().mean() / (1024 * 1024)
      ),
  }


def analyze_temporal_locality(
    df: pd.DataFrame, block_size_bytes: int = 4096
) -> Dict[str, float]:
  """計算時間局部性指標（以 LBA 邏輯區塊為基準）

  :param df: 包含 'Offset' 欄位的 DataFrame
  :param block_size_bytes: 邏輯區塊大小 (預設 4KB = 4096 Bytes)
  """
  # 將 Byte 偏移量映射到 Block ID
  block_ids = df["Offset"] // block_size_bytes
  total_requests = len(block_ids)
  unique_blocks = block_ids.nunique()

  # 唯一區塊比例越低，代表重複存取同一個 Block 的頻率越高（時間局部性越強）
  unique_ratio = unique_blocks / total_requests if total_requests > 0 else 0.0

  return {
      "total_requests": total_requests,
      "unique_blocks_count": int(unique_blocks),
      "unique_block_ratio_%": float(unique_ratio * 100),
      "block_reuse_ratio_%": float((1.0 - unique_ratio) * 100),
  }


# ==============================================================================
# 模組二：磁碟排程演算法 (FCFS, SSTF, LOOK)
# ==============================================================================


class DiskScheduler:

  def __init__(self, requests: List[int], initial_head: int):
    """初始化排程器

    :param requests: 請求的區塊或磁軌清單 (整數型態，如 Block ID 或 Cylinder ID)
    :param initial_head: 磁頭起始位置
    """
    self.requests = list(requests)
    self.initial_head = initial_head

  def fcfs(self) -> Tuple[List[int], int]:
    """FCFS (First-Come, First-Served) 先到先服務

    按照佇列原始進入順序處理，最公平、無飢餓風險，但移動距離可能極長。
    """
    current_head = self.initial_head
    total_movement = 0
    service_order = []

    for req in self.requests:
      service_order.append(req)
      total_movement += abs(req - current_head)
      current_head = req

    return service_order, total_movement

  def sstf(self) -> Tuple[List[int], int]:
    """SSTF (Shortest Seek Time First) 最短尋道時間優先

    每次選擇距離目前磁頭最近的請求處理（Greedy 策略）。
    尋道開銷顯著降低，但在高負載下易引發飢餓現象 (Starvation)。
    """
    current_head = self.initial_head
    total_movement = 0
    service_order = []
    pending = self.requests.copy()

    while pending:
      # 找出與目前磁頭距離最小的項目
      closest_req = min(pending, key=lambda r: abs(r - current_head))
      service_order.append(closest_req)
      total_movement += abs(closest_req - current_head)
      current_head = closest_req
      pending.remove(closest_req)

    return service_order, total_movement

  def look(
      self, direction: Literal["up", "down"] = "up"
  ) -> Tuple[List[int], int]:
    """LOOK (電梯演算法變形)

    磁頭朝單一方向前進並服務沿途請求，抵達該方向的最末端請求後才掉頭。
    有效平衡總尋道距離與避免極端磁軌飢餓。
    :param direction: 初始移動方向，'up' (向外/向上) 或 'down' (向內/向下)
    """
    current_head = self.initial_head
    total_movement = 0
    service_order = []

    # 依當前磁頭位置分群並排序
    greater = sorted([r for r in self.requests if r >= current_head])
    smaller = sorted(
        [r for r in self.requests if r < current_head], reverse=True
    )

    if direction == "up":
      first_leg, second_leg = greater, smaller
    else:
      first_leg, second_leg = smaller, greater

    for req in first_leg + second_leg:
      service_order.append(req)
      total_movement += abs(req - current_head)
      current_head = req

    return service_order, total_movement


# ==============================================================================
# 模組三：批次測試與模擬驗證範例
# ==============================================================================

if __name__ == "__main__":
  # 1. 建立測試資料（結構等同於 MSR Cambridge Trace）
  # 實際使用時，改為: df = pd.read_csv("your_trace_file.csv", names=['Timestamp', 'Hostname', 'DiskNumber', 'Type', 'Offset', 'Size', 'ResponseTime'])
  demo_data = {
      "Timestamp": [1001, 1005, 1009, 1015, 1020, 1028, 1035, 1042, 1050],
      "Type": [
          "Write",
          "Write",
          "Read",
          "Write",
          "Read",
          "Read",
          "Write",
          "Read",
          "Write",
      ],
      "Offset": [
          40960,  # 區塊 10
          45056,  # 區塊 11 (緊接著前一個: 40960 + 4096 = 45056 -> 連續)
          49152,  # 區塊 12 (緊接著前一個: 45056 + 4096 = 49152 -> 連續)
          409600,  # 區塊 100 (遠端隨機跳轉)
          413696,  # 區塊 101 (連續)
          40960,  # 區塊 10 (重複存取 -> 時間局部性)
          819200,  # 區塊 200 (跳轉)
          823296,  # 區塊 201 (連續)
          49152,  # 區塊 12 (重複存取)
      ],
      "Size": [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096],
  }
  df = pd.DataFrame(demo_data)

  print("=" * 60)
  print("一、Locality（局部性）特徵計算結果")
  print("=" * 60)

  spatial_results = analyze_spatial_locality(df, threshold_bytes=65536)
  for k, v in spatial_results.items():
    print(f"{k:<30}: {v:.2f}")

  print("-" * 60)
  temporal_results = analyze_temporal_locality(df, block_size_bytes=4096)
  for k, v in temporal_results.items():
    print(f"{k:<30}: {v:.2f}")

  print("\n" + "=" * 60)
  print("二、磁碟排程模擬 (以 4KB 區塊編號為磁軌目標)")
  print("=" * 60)

  # 將 Offset 轉換為 Block ID 做為磁碟請求序列
  blocks = (df["Offset"] // 4096).tolist()
  initial_head = blocks[0]  # 假設以第一個請求為磁頭初始起點
  request_queue = blocks[1:]  # 後續抵達佇列的請求

  scheduler = DiskScheduler(requests=request_queue, initial_head=initial_head)

  fcfs_order, fcfs_dist = scheduler.fcfs()
  sstf_order, sstf_dist = scheduler.sstf()
  look_order, look_dist = scheduler.look(direction="up")

  print(f"磁頭起點 (Initial Head)   : {initial_head}")
  print(f"待處理請求佇列 (Block IDs) : {request_queue}\n")

  print(f"1. FCFS 服務順序 : {fcfs_order}")
  print(f"   總移動距離    : {fcfs_dist} 個區塊")

  print(f"2. SSTF 服務順序 : {sstf_order}")
  print(
      f"   總移動距離    : {sstf_dist} 個區塊 (較 FCFS 節省"
      f" {((fcfs_dist - sstf_dist) / fcfs_dist) * 100:.1f}%)"
  )

  print(f"3. LOOK 服務順序 : {look_order}")
  print(
      f"   總移動距離    : {look_dist} 個區塊 (較 FCFS 節省"
      f" {((fcfs_dist - look_dist) / fcfs_dist) * 100:.1f}%)"
  )
  print("=" * 60)
```

---

### 如何套用到你的真實 Trace 檔案？

當你從 GitHub 網站下載了 MSR Cambridge 的真實檔案（例如 `src1_0.csv` 或 `usr_0.csv`），只需將程式碼第 155 行替換為：

```python
# 讀取真實的 MSR Cambridge Trace 檔案
columns = [
    'Timestamp',
    'Hostname',
    'DiskNumber',
    'Type',
    'Offset',
    'Size',
    'ResponseTime',
]
df = pd.read_csv('src1_0.csv', header=None, names=columns)

# 由於原始資料量通常有數百萬筆，排程模擬建議取前 1,000 或 5,000 筆作為代表性樣本窗口
sample_df = df.head(1000)
```
