import math
from typing import List
class Solution1:
    def coinChange_wrong(self, coins: List[int], amount: int) -> int:
        if amount == 0:
            return 0
        coins = sorted(coins, reverse=True)
        coins_count = -1
        for i in range(len(coins)):
            c = coins[i]
            if amount == 0:
                break
            if c > amount:
                continue

            c_count = int(math.floor(amount / c))
            coins_count += c_count + (coins_count == -1)
            amount -= c_count * c
        if amount > 0:
            return -1
        return coins_count

    def coinChange_recur(self, coins: List[int], amount: int, ind: int):
        c = coins[ind]
        if amount == 0:
            return 0
        if amount % c == 0:
            print(amount//c, c)
            return amount // c
        if (ind == len(coins) - 1):
            return -1
        for i in range(amount // c, -1, -1):
            n = self.coinChange_recur(coins, amount - c * i, ind + 1)
            if n != -1:
                print(i,c)
                return n + i
        return -1

    def coinChange(self, coins: List[int], amount: int) -> int:
        new_coins = sorted(coins, reverse=True)
        return self.coinChange_recur(new_coins, amount, 0)


class Solution:
    def coinChange_recur(self, coins: List[int], amount: int, ind: int):
        c = coins[ind]
        if amount == 0:
            return 0
        if amount % c == 0:
            return amount // c
        if (ind == len(coins) - 1):
            return -1
        min_c = -1
        for i in range(amount // c, -1, -1):
            n = self.coinChange_recur(coins, amount - c * i, ind + 1)
            if n != -1:
                min_c = n + i if min_c == -1 else min(n + i, min_c)
        return min_c


coins = [186,419,83,408]
amount = 6249
expected = 20
print(Solution().coinChange(coins, amount))

coins = sorted(coins, reverse=True)
def subsetSum_gen(dim, n):
    if n==0:
        yield [0]*dim
        return
    if dim==1:
        yield [n]
        return
    for i in range(n,-1,-1):
        for s in subsetSum_gen(dim-1, n-i):
            yield [i]+s
for s in subsetSum_gen(4,20):
    if s[0]*coins[0] + s[1]*coins[1] + s[2]*coins[2] + s[3]*coins[3]==amount:
        print(s, ":", coins)

for i in range(3):
    for j in range(i+1,4):
        print(coins[i], coins[j], math.gcd(coins[i], coins[j]))

#minimal counter example:
#4,3,1 to make 6 - 4 1 1 instead of 3 3
#so we HAVE to recur like this.