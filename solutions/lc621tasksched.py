from collections import Counter, deque
from typing import List
import heapq
#https://leetcode.com/problems/task-scheduler/submissions/1598870680/
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        task_heap = []
        task_cooldown = deque()
        task_counts = Counter(tasks)
        task_num = len(task_counts.keys())
        task_num_comp = 0
        break_time = len(tasks) * (n+2)
        # neg_count, letter on heap
        # cooldown - queue of letter, neg count, time it's cool
        time = 0
        for t, c in task_counts.items():
            heapq.heappush(task_heap, (-1 * c, t))

        # pop task off if there is one
        # queue
        while True:
            while task_cooldown:
                if task_cooldown[0][2] == time:
                    t, n_c, _ = task_cooldown.popleft()
                    heapq.heappush(task_heap, (n_c, t))
                else:
                    break
            if task_heap:
                n_c, t = heapq.heappop(task_heap)
                n_c += 1
                if n_c < 0:
                    task_cooldown.append((t, n_c, time + n + 1))
                else:
                    task_num_comp += 1
            if task_num == task_num_comp:
                return time+1
            if time > break_time:
                return break_time
            time += 1
        return -1
s = Solution()
tasks = ["A","A","A","B","B","B"]
n = 2
print(s.leastInterval(tasks, n))
#8
tasks = ["A","C","A","B","D","B"]
n=1
print(s.leastInterval(tasks, n))
#6
tasks=["A","A","A", "B","B","B"]
n=3
print(s.leastInterval(tasks, n))
#10
