from typing import List
class Solution:
    def change_minus_one(self, amount: int, coins: List[int]) -> int:
        dp = [1] + [-1]*amount
        for c in coins:
            for i in range(1,amount+1):
                if (i-c)>=0:
                    if dp[i-c]!=-1 and dp[i]==-1:
                        dp[i]=dp[i-c]
                    elif dp[i-c]!=-1 and dp[i]!=-1:
                        dp[i]+=dp[i-c]
        return dp[amount]

    def change(self, amount: int, coins: List[int]) -> int:
        dp = [1] + [0]*amount
        for c in coins:
            for i in range(1,amount+1):
                if (i-c)>=0:
                    dp[i]+=dp[i-c]
        return dp[amount]
s=Solution()
print(s.change(3,[2]))