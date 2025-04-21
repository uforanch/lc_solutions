from collections import deque, Counter
import math

class Solution:
    def minWindow(self, t: str, s: str) -> str:
        """
        first idea, deque of char locations
        move pointer when it's possible to
        """
        s_counter = Counter(s)
        t_counter = Counter()
        eq_count = 0
        gt_count = 0
        eq_max = len(s_counter.keys())
        min_window=""
        min_length=math.inf
        t_deque = deque()
        p1=None
        for p2 in range(1,len(t)+1):
            c = t[p2-1]
            if s_counter[c]==0:
                continue
            if p1 is None:
                p1 = p2-1
            t_deque.append((c,p2-1))
            if t_counter[c]==s_counter[c]:
                eq_count-=1
                gt_count+=1
            t_counter[c]+=1
            if t_counter[c]==s_counter[c]:
                eq_count+=1
            elif t_counter[c]>s_counter[c]:
                while len(t_deque)>0:
                    ch, ph = t_deque[0]
                    p1=ph
                    if t_counter[ch] > s_counter[ch]:
                        t_deque.popleft()
                        t_counter[ch]-=1
                        if t_counter[ch]==s_counter[ch]:
                            gt_count-=1
                            eq_count+=1
                    else:
                        break
            if eq_count+gt_count==eq_max:
                if p2-p1 < min_length:
                    min_length = p2-p1
                    min_window = t[p1:p2]


        return min_window


sol = Solution()
#print(s.minWindow("ADOBECODEBANC", "ABC"))


s="aaaaaaaaaaaabbbbbcdd"
t="abcdd"
print(sol.minWindow(s,t))