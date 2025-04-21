from functools import cache
import math
"""
read solution but keeping this for lesson
I cannot BELIEVE naive recursion + "@cache" is good enough here

then did it as a table exercise... forgot to make sure case was in ELSE

"""
#https://leetcode.com/problems/edit-distance/
class Solution:
    @cache
    def minDistance_cache(self, word1: str, word2: str) -> int:
        if len(word1)==0 or len(word2)==0:
            return max(len(word1), len(word2))
        if word1[0]==word2[0]:
            return self.minDistance(word1[1:], word2[1:])
        insert = 1 + self.minDistance(word1,word2[1:])
        delete = 1 + self.minDistance(word1[1:], word2)
        replace = 1 + self.minDistance(word1[1:],word2[1:])
        return min(insert, delete, replace)


    def minDistance(self, word1: str, word2: str) -> int:
        if len(word1) == 0 or len(word2) == 0:
            return max(len(word1), len(word2))
        n = len(word1)
        m = len(word2)
        dp = [[math.inf for _ in range(m + 1)] for _ in range(n + 1)]
        # dp[i][j] = edit from word1[:i] to word2[:j]
        # which is then... sort of the same thing we were doing
        # but check last character
        for r in range(n + 1):
            dp[r][0] = r
        for c in range(m + 1):
            dp[0][c] = c
        for r in range(1, n + 1):
            for c in range(1, m + 1):
                if word1[r - 1] == word2[c - 1]:
                    dp[r][c] = dp[r - 1][c - 1]
                else:
                    dp[r][c] = 1 + min(dp[r][c - 1], dp[r - 1][c], dp[r - 1][c - 1])
        return dp[n][m]
s=Solution()
print(s.minDistance("horse","ros"))#3
print(s.minDistance("intention", "execution"))#5
print(s.minDistance("xxx","xxx"))