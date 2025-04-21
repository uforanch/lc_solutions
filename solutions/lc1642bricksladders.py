from typing import List
import math, heapq


class Solution:
    def furthestBuilding(self, heights: List[int], bricks: int, ladders: int) -> int:
        """
        keep #ladder max diffs as we go forward
        when remaining diffs are > bricks we're done

        min heap GOOD: we want to get the smallest out
        """
        max_building = 0
        bricks_used = 0
        k_ladders = []
        for i0 in range(1, len(heights)):
            if heights[i0] <= heights[i0 - 1]:
                max_building = i0
                continue
            h = heights[i0] - heights[i0 - 1]
            heapq.heappush(k_ladders, h)
            if len(k_ladders) > ladders:
                h_pop = heapq.heappop(k_ladders)
                bricks_used += h_pop
                if bricks_used>bricks:
                    return max_building
            max_building=i0
        return max_building
s=Solution()
# print(s.furthestBuilding([4,2,7,6,9,14,12], 4, 1))#4
# print(s.furthestBuilding([4,12,2,7,3,18,20,3,19], 10, 2))#7
# print(s.furthestBuilding([14,3,19,3], 17, 0))#3
print(s.furthestBuilding([3,19], 87, 1))#3
