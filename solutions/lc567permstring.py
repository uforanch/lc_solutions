from collections import Counter
#https://leetcode.com/problems/permutation-in-string/description/

class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        s1_count = Counter(s1)
        s2_count = Counter()
        eq_count = 0
        eq_max = len(s1_count.keys())
        p1 = 0
        # counting s2[p1:p2]
        for p2 in range(1, len(s2) + 1):
            s2_count[s2[p2 - 1]] += 1
            if s2_count[s2[p2 - 1]] == s1_count[s2[p2 - 1]]:
                eq_count += 1
            if eq_count == eq_max:
                return True
            if s2_count[s2[p2 - 1]] > s1_count[s2[p2 - 1]]:
                for i0 in range(p1, p2 + 1):
                    if s2_count[s2[i0]] == s1_count[s2[i0]]:
                        eq_count -= 1
                    s2_count[s2[i0]] -= 1
                    p1 = i0 + 1
                    if s2_count[s2[p2 - 1]] <= s1_count[s2[p2 - 1]]:
                        break
        return False








