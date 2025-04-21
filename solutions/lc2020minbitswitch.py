class Solution:
    def minBitFlips(self, start: int, goal: int) -> int:
        n = max(start, goal)
        m = min(start, goal)
        bit_sum = 0
        while n > 0:
            n_b = n % 2
            m_b = m % 2
            bit_sum += (m_b != n_b)
            n = (n - n_b) / 2
            m = (m - m_b) / 2

        return bit_sum

    def minBitFlips_dl(self, start: int, goal: int) -> int:
        # XOR to find differing bits
        xor_result = start ^ goal
        count = 0
        # Count the number of 1s in xor_result (differing bits)
        while xor_result:
            count += xor_result & 1  # Increment if the last bit is 1
            xor_result >>= 1  # Shift right to process the next bit
        return count