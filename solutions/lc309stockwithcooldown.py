from typing import List
#https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        """
        brutish force with dynamic programming first
        dp[i] - max profit if at i is doing nothing or cooldown

        """
        dp = [0 for _ in range(len(prices)+1)]
        for i0 in range(1,len(prices)+1):
            #doing nothing - use last problem if possible
            dp[i0] = 0 if i0==0 else dp[i0-1]
            #cooldown - last action was sell
            for i1 in range(0,i0-1):
                c = 0 if i1==0 else dp[i1-1]
                dp[i0] = max(dp[i0], c+prices[i0-1]-prices[i1])
        return dp[len(prices)]
