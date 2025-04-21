from typing import List
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals_sorted = intervals.copy()
        intervals_sorted.sort()
        last_interval = intervals_sorted[0]
        new_intervals = []
        for interval in intervals_sorted:
            if last_interval[1] >= interval[0]:
                # merge
                last_interval[1] = max(last_interval[1], interval[1])
            else:
                new_intervals.append(last_interval)
                last_interval = interval
        if last_interval not in new_intervals:
            new_intervals.append(last_interval)
        return new_intervals

print(Solution().merge([[1,4],[0,4]]))