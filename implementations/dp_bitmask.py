from typing import List
from itertools import product
"""
the subset sum problem in materials sent to me 
has straight up incorrect code for subset sum 

I'll do leetcode with bitmask instead
"""


class Solution:
    #https://leetcode.com/problems/single-number/
    def singleNumber(self, nums: List[int]) -> int:
        bits = 0
        for n in nums:
            bits = bits ^ n
        return bits
    #https://leetcode.com/problems/number-of-1-bits/submissions/1598875313/
    #BEATS 100%
    def hammingWeight(self, n: int) -> int:
        bits = 0
        while n > 0:
            bits += n % 2
            n = n >> 1
        return bits
    #https://leetcode.com/problems/reverse-bits/submissions/1598880287/
    class Solution:
        def reverseBits(self, n: int) -> int:
            out = 0
            for _ in range(32):
                out = out << 1
                out += n % 2
                n = n >> 1
            return out
    #https://leetcode.com/problems/missing-number/
    #they sure like xor trick
    def missingNumber(self, nums: List[int]) -> int:
        bits=0
        for i in range(len(nums)+1):
            bits = bits^i
        for n in nums:
            bits = bits^n
        return bits

    #https://leetcode.com/problems/sum-of-two-integers/
    #needcode solution
    def getSum(self, a: int, b: int) -> int:
        """
        mask - keeps things to 32 bits
        max_int - max POSITIVE int, ONE bit short
        (a&m)^(b&m)==(a^b)&m

        two's compliment it's not just the super bit but EVERY bit
        -2&mask = 0b111.....11
        so what we're actually doing with negatives is basically
        taking mask-neg and adding, then taking result mod mask

        probably going to have to memorize ths because I don't entirely get it

        :param a:
        :param b:
        :return:
        """
        mask = 0xFFFFFFFF
        max_int = 0x7FFFFFFF

        while b != 0:
            carry = (a & b) << 1
            a = (a ^ b) & mask
            b = carry & mask

        return a if a <= max_int else ~(a ^ mask)
    def test_expression(self, func, dim):
        for i in range(2**dim):
            entry = [b=="1" for b in format(i, f'0{dim}b')]
            print(entry, func(*entry))

s = Solution()
print(s.getSum(-2,3))
s.test_expression(lambda a,b,c : (a&c)^(b&c)==(a^b)&c, 3)