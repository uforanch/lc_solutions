from typing import List
import math
class Solution:
    def coinChange_recur(self, coins: List[int], amount: int, ind: int, total_coins: int):
        c = coins[ind]
        if amount == 0:
            return True
        if amount > total_coins * c:
            return False
        if ind == len(coins) - 1:
            return amount == (total_coins * c)
        for i in range(total_coins, -1, -1):
            if self.coinChange_recur(coins, amount - i * c, ind + 1, total_coins - i):
                return True
        return False

    def coinChange_wrong(self, coins: List[int], amount: int) -> int:
        if amount==0:
            return 0
        new_coins = sorted(coins, reverse=True)
        min_coins = amount // new_coins[0]
        max_coins = amount // new_coins[-1]
        for total_coins in range(min_coins, max_coins + 1):
            if self.coinChange_recur(new_coins, amount, 0, total_coins):
                return total_coins
        return -1

    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [amount + 1] * (amount + 1)
        dp[0] = 0  # Base case: 0 coins to make amount 0

        for i in range(1, amount + 1):
            for coin in coins:
                if i - coin >= 0:
                    dp[i] = min(dp[i], dp[i - coin] + 1)

        return dp[amount] if dp[amount] <= amount else -1

    def coinChange_alt(self, coins: List[int], amount: int) -> int:
        dp = [0] + [math.inf]*amount
        coins.sort()
        for c in coins:
            for i in range(1,amount+1):
                if i-c>=0:
                    dp[i] = min(1+dp[i-c], dp[i])
        return dp[amount] if dp[amount] != math.inf else -1


s=Solution()
#print(s.coinChange([1,2,5],11))
print(s.coinChange([2,5,10,1],27))