"""
题目三（Optional）：高性能滑动窗口统计
======================================
实现 SlidingWindowStats 类:
  - __init__(window_sec)      — 设置窗口秒数
  - add(timestamp, value)     — 添加数据点，自动淘汰过期数据（均摊 O(1)）
  - get_mean()                — O(1) 返回均值，空窗口返回 0.0
  - get_max()                 — O(1) 返回最大值，空窗口返回 0.0

约束:
  - 每个数据点只存一次
  - 禁止用 pandas/numpy/scipy，只能用 collections.deque
  - get_max 不能遍历所有元素（用单调队列）

单调队列（Monotonic Queue）原理:
    max_q 是一个单调递减的双端队列，队首始终是窗口内的最大值。
    当新值到来时:
      1. 从队尾弹出所有 ≤ 新值的元素（它们不可能成为最大值了）。
      2. 将新值和 timestamp 加入队尾。
    当淘汰过期数据时:
      3. 如果队首的 timestamp 过期，从队首弹出（popleft）。
    这样队首始终是窗口内的最大值，get_max 直接返回 max_q[0] 即可。

均值维护:
    total 维护窗口内所有值的和，count 维护元素个数。
    每次 add 时: total += value, count += 1
    淘汰数据时: total -= 被淘汰的 value, count -= 1
    get_mean 直接返回 total / count（注意 count==0 时返 0.0）。
"""

from collections import deque


class SlidingWindowStats:
    """基于单调队列的滑动窗口统计，get_mean / get_max 均为 O(1)。

    Attributes:
        window_sec: 窗口宽度（秒）。
        data: 存储所有窗口内数据点的 deque，元素为 (timestamp, value)。
        max_q: 单调递减队列，元素为 (timestamp, value)，队首即当前窗口最大值。
        total: 窗口内所有值的和，用于 O(1) 计算均值。
        count: 窗口内数据点个数。
    """

    def __init__(self, window_sec: float):
        """初始化滑动窗口。

        Args:
            window_sec: 窗口宽度（秒）。窗口包含 [t - window_sec, t] 内的所有点。
        """
        self.window_sec = window_sec
        self.data = deque()    # 存所有窗口内 (timestamp, value)
        self.max_q = deque()   # 单调队列，存 (timestamp, value)，单调递减
        self.total = 0.0
        self.count = 0

    def add(self, timestamp: float, value: float) -> None:
        """添加一个数据点，并自动淘汰窗口外的过期数据。

        执行步骤:
          1. 将 (timestamp, value) 加入 data 队尾。
          2. 从 data 队首弹出所有 timestamp < timestamp - window_sec 的数据。
          3. 更新 total 和 count（加新增、减淘汰）。
          4. 维护 max_q 单调递减:
             a. 从 max_q 队尾弹出所有 value <= 新值的元素（它们被新值"压制"了）。
             b. 将新值加入 max_q 队尾。
             c. 从 max_q 队首弹出所有 timestamp 过期的元素。

        均摊 O(1): 虽然单次循环可能多次弹出，但每个元素最多入队一次、出队一次。

        Args:
            timestamp: 数据点的时间戳（秒）。
            value: 数据点的值。
        """
        pass

    def get_mean(self) -> float:
        """返回窗口内所有值的算术平均，O(1)。

        Returns:
            均值；如果窗口为空返回 0.0。
        """
        pass

    def get_max(self) -> float:
        """返回窗口内的最大值，O(1)。

        直接读取 max_q 队首元素的值。
        注意: 如果 max_q 和 data 维护正确，队首始终是当前窗口内最大值。

        Returns:
            最大值；如果窗口为空返回 0.0。
        """
        pass


if __name__ == "__main__":
    # 题目给出的测试用例
    stats = SlidingWindowStats(window_sec=5.0)
    stats.add(0.0, 10)
    stats.add(1.0, 20)
    stats.add(2.0, 30)
    print(stats.get_mean())  # 预期: 20.0
    print(stats.get_max())   # 预期: 30.0

    stats.add(6.0, 40)  # timestamp < 1.0 的数据过期（0.0 的 10 被淘汰）
    print(stats.get_mean())  # 预期: 30.0  (20+30+40)/3
    print(stats.get_max())   # 预期: 40

    # TODO: 补充更多边界测试用例，例如:
    # - 空窗口: get_mean/get_max 应返回 0.0
    # - 相同值: 多个相等的 value，单调队列不会丢失数据
    # - 乱序时间戳: 如果 timestamp 不单调递增，行为是否正确？
    # - 单个元素: 窗口内只有一个值的场景
