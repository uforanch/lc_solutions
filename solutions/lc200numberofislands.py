from typing import  List
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:

        explored = [[0 for _ in row] for row in grid]
        rows = len(explored)
        cols = len(explored[0])
        components = 0

        for r_i, row in enumerate(grid):
            for c_i, entry in enumerate(row):
                if explored[r_i][c_i] == 1 or entry == "0":
                    continue
                    # entry is 1 and uneplored
                components += 1
                stack = [(r_i, c_i)]
                print("*")
                while stack:
                    r, c = stack.pop()
                    new_stack = []
                    print(r,c)
                    if r - 1 >= 0:
                        new_stack.append((r - 1, c))
                    if r + 1 <= rows - 1:
                        new_stack.append((r + 1, c))
                    if c - 1 >= 0:
                        new_stack.append((r, c - 1))
                    if c + 1 <= cols - 1:
                        new_stack.append((r, c + 1))
                    for (r0, c0) in new_stack:
                        if grid[r0][c0] == "1" and (r0, c0) not in stack and explored[r0][c0] == 0:
                            stack.append((r0, c0))
                    explored[r][c] = 1
        return components


tc=[["1","1","1"],["0","1","0"],["1","1","1"]]
print(Solution().numIslands(tc))