from collections import defaultdict
from typing import List
import math
import heapq
class Solution:
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        dist = [[math.inf for _ in range(len(heights[0]))] for _ in range(len(heights))]
        dist[0][0] = 0
        Q = [(0, 0, 0)]
        rows = len(heights)
        cols = len(heights[0])
        unvisited = set()
        for r in range(rows):
            for c in range(cols):
                unvisited.add((r, c))

        while unvisited:
            eff, r, c = heapq.heappop(Q)
            if (r, c) not in unvisited:
                continue
            for rn, cn in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
                if rn < 0 or rn >= rows or cn < 0 or cn >= cols:
                    continue
                if (rn, cn) not in unvisited:
                    continue
                dist[rn][cn] = min(dist[rn][cn], max(dist[r][c], abs(heights[r][c] - heights[rn][cn])))
                heapq.heappush(Q, (dist[rn][cn], rn, cn))
            #print(r,c, unvisited)
            unvisited.remove((r, c))

        return dist[rows - 1][cols - 1]


    def minimumEffortPath_orig(self, heights: List[List[int]]) -> int:
        effort = [[math.inf for _ in heights[i]] for i in range(len(heights))]
        prev = [[-1 for _ in heights[i]] for i in range(len(heights))]
        effort[0][0] = 0
        val_to_item_sets = defaultdict(set)
        Q = set()
        for r in range(len(heights)):
            for c in range(len(heights[r])):
                val_to_item_sets[effort[r][c]].add((r, c))
                Q.add((r, c))
        while Q:
            # get from queue
            item = None
            while True:
                v = min(val_to_item_sets.keys())
                if not val_to_item_sets[v]:
                    del val_to_item_sets[v]
                else:
                    item = list(val_to_item_sets[v])[0]
                    val_to_item_sets[v].remove(item)
                    Q.remove(item)
                    break
            r, c = item
            for rr, cc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
                if (rr, cc) not in Q:  # vertex exists and is in Q
                    continue
                #alt = effort[r][c] + abs(heights[r][c] - heights[rr][cc])
                alt = max(effort[r][c], abs(heights[r][c] - heights[rr][cc]))
                if alt < effort[rr][cc]:
                    e = effort[rr][cc]
                    val_to_item_sets[e].remove((rr, cc))
                    val_to_item_sets[alt].add((rr, cc))
                    effort[rr][cc] = alt
        print(effort)
        return effort[len(heights) - 1][len(heights[0]) - 1]

    def minimumEffortPath_dl(self, heights):

        if not heights:
            return 0

        rows, cols = len(heights), len(heights[0])

        min_heap = [(0, 0, 0)]  # (effort, row, col)

        max_effort = 0

        visited = set()

        while min_heap:

            effort, cur_row, cur_col = heapq.heappop(min_heap)

            max_effort = max(max_effort, effort)

            if (cur_row, cur_col) == (rows - 1, cols - 1):
                return max_effort

            visited.add((cur_row, cur_col))

            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:

                new_row, new_col = cur_row + dr, cur_col + dc

                if 0 <= new_row < rows and 0 <= new_col < cols and (new_row, new_col) not in visited:
                    new_effort = abs(heights[new_row][new_col] - heights[cur_row][cur_col])

                    heapq.heappush(min_heap, (new_effort, new_row, new_col))

        return max_effort

s = Solution()
print(s.minimumEffortPath([[1,2,2],[3,8,2],[5,3,5]])) # 2
print(s.minimumEffortPath([[1,2,3],[3,8,4],[5,3,5]])) # 1
print(s.minimumEffortPath([[1,2,1,1,1],[1,2,1,2,1],[1,2,1,2,1],[1,2,1,2,1],[1,1,1,2,1]])) #0
print(s.minimumEffortPath([[1,2],[4,3]]))

