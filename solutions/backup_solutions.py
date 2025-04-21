from typing import List, Optional
from collections import defaultdict, Counter, dequegi
from itertools import product
from base import TreeNode, ListNode
import math

def listprod(nums):
    #productExceptSelf
    notzero = lambda x : 1 if x==0 else x
    prod=1
    for n in nums:
        prod*=notzero(n)
    return prod

class Solution:
    #ARRAYS AND HASHING
    def containsDuplicate(self, nums: List[int]) -> bool:
        #https://leetcode.com/problems/contains-duplicate/
        s=set()
        for i in nums:
            if i in s:
                return True
            s.add(i)
        return False

    def isAnagram(self, s: str, t: str) -> bool:
        #https://leetcode.com/problems/valid-anagram/
        return sorted(s)==sorted(t)

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        #https://leetcode.com/problems/two-sum/
        numdict = dict()
        for i, n in enumerate(nums):
            if target-n in numdict.keys():
                return [numdict[target-n],i]
            numdict[n]=i

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        #https://leetcode.com/problems/group-anagrams/
        dd = defaultdict(list)
        for s in strs:
            dd["".join(sorted(s))].append(s)
        return [v for v in dd.values()]

    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        #https://leetcode.com/problems/top-k-frequent-elements/
        kstack = []
        xtrastack = []
        count_dict = defaultdict(lambda: 0)
        for n in nums:
            count_dict[n] += 1
            if n not in kstack:
                if len(kstack) < k:
                    kstack.append(n)
                elif count_dict[kstack[-1]] < count_dict[n]:
                    while len(kstack) > 0:
                        if count_dict[kstack[-1]] < count_dict[n]:
                            xtrastack.append(kstack.pop(-1))
                        else:
                            break
                    kstack.append(n)
                    while len(kstack) < k:
                        kstack.append(xtrastack.pop(-1))
                    xtrastack = []
            else:
                while kstack[-1] != n:
                    xtrastack.append(kstack.pop(-1))
                kstack.pop(-1)
                while len(kstack) > 0:
                    if count_dict[kstack[-1]] < count_dict[n]:
                        xtrastack.append(kstack.pop(-1))
                    else:
                        break
                kstack.append(n)
                while len(xtrastack) > 0 and len(kstack) <= k:
                    kstack.append(xtrastack.pop(-1))
                xtrastack = []

        return kstack

    def productExceptSelf(self, nums: List[int]) -> List[int]:
        #https://leetcode.com/problems/product-of-array-except-self/
        zero_count = nums.count(0)
        if zero_count == 0:
            prod = listprod(nums)
            return [int(prod/n) for n in nums]
        elif zero_count>1:
            return [0 for _ in range(len(nums))]
        else:
            prod = listprod(nums)
            return [prod if n==0 else 0 for n in nums]

    def isValidSudoku(self, board: List[List[str]]) -> bool:
        #https://leetcode.com/problems/valid-sudoku/
        checkareas=[]
        #rows
        for i in range(9):
            checkareas.append(list(product([i], list(range(9)))))
        #cols
        for i in range(9):
            checkareas.append(list(product(list(range(9)),[i])))
        #cells
        for i in range(0,9,3):
            for j in range(0,9,3):
                checkareas.append(list(product([i,i+1,i+2],[j,j+1,j+2])))

        for pl in checkareas:
            a_count = Counter([board[r][c] for r,c in pl if board[r][c]!="."])
            if len(a_count.values())>0:
                if max(a_count.values())>1:
                    return False
        return True

    class Solution:
        def longestConsecutive(self, nums: List[int]) -> int:
            #https://leetcode.com/problems/longest-consecutive-sequence/
            length_dict = {}
            unsearched = set(nums)
            m_l = 0
            while unsearched:
                n = unsearched.pop()
                k = n
                while k + 1 in unsearched:
                    k = k + 1
                    unsearched.remove(k)
                if k + 1 in length_dict.keys():
                    ll = length_dict[k + 1]
                else:
                    ll = 0
                    length_dict[k + 1] = 0
                for i in range(n, k + 1):
                    length_dict[i] = k + 1 - n + ll
                if length_dict[n] > m_l:
                    m_l = length_dict[n]
            return m_l

    #TWO POINTERS
    def isPalindrome(self, s: str) -> bool:
        #https://leetcode.com/problems/valid-palindrome/
        x = set([chr(i) for i in range(ord('a'), ord('z')+1)]).union(set([chr(i) for i in range(ord('0'), ord('9')+1)]))
        s1 = "".join([a for a in s.lower() if a in x])
        for i in range(len(s1)):
            if s1[i] != s1[len(s1)-1-i]:
                return False
        return True

    def twoSum_II(self, numbers: List[int], target: int) -> List[int]:
        #function is just twoSum in problem
        #https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/
        i=0
        j=len(numbers)-1
        while True:
            if numbers[i]+numbers[j]>target:
                j=j-1
            elif numbers[i]+numbers[j]<target:
                i=i+1
            elif numbers[i]+numbers[j]==target:
                return [i+1,j+1]

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        #https://leetcode.com/problems/3sum/
        nums_set = set(nums)
        nums_sort = sorted(list(nums_set))
        nums_count = Counter(nums)
        triples = []
        for i, s in enumerate(nums_sort):
            if s > 0:
                break
            if s == 0:
                if nums_count[s] >= 3:
                    triples.append([0, 0, 0])
                break
            if nums_count[s] > 1 and -2 * s in nums_set:
                triples.append([s, s, -2 * s])
            if (-1 * s) % 2 == 0:
                j = int((-1 * s) / 2)
                if j in nums_set and nums_count[j] > 1:
                    triples.append([s, j, j])

            for j in range(i + 1, len(nums_sort)):
                s2 = nums_sort[j]
                if s + s2 > 0:
                    break
                s3 = -1 * (s + s2)
                if s3 > s2 and s3 in nums_set:
                    triples.append([s, s2, s3])
        return triples

    def maxArea(self, height: List[int]) -> int:
        #https://leetcode.com/problems/container-with-most-water/
        h_list = sorted([(h, i) for i, h in enumerate(height)])
        h_list.reverse()
        n = len(height)

        l_side = min(h_list[0][1], h_list[1][1])
        r_side = max(h_list[0][1], h_list[1][1])

        m_vol = h_list[1][0] * abs(h_list[1][1] - h_list[0][1])

        for i in range(2, n):
            place = h_list[i][1]
            max_base = max(abs(place - l_side), abs(r_side - place))
            m_vol = max(max_base * h_list[i][0], m_vol)
            l_side = min(l_side, place)
            r_side = max(r_side, place)
        return m_vol

    def trap(self, height: List[int]) -> int:
        #https://leetcode.com/problems/trapping-rain-water/
        # first get max and all indices of the max
        n = len(height)
        m = max(height)
        m_indices = [i for i, v in enumerate(height) if v == m]
        # do the ends
        l_m_ind = m_indices[0]
        r_m_ind = m_indices[-1]

        total_w_cont = 0
        for r in [range(0, l_m_ind), range(n - 1, r_m_ind, -1)]:

            l_val = None
            for i in r:
                h = height[i]
                if l_val is None:
                    l_val = h
                elif h > l_val:
                    l_val = h
                else:
                    total_w_cont += (l_val - h)

        for i in range(l_m_ind, r_m_ind):
            total_w_cont += (m - height[i])
        return total_w_cont

    #STACK
    def isValid(self, s: str) -> bool:
        #https://leetcode.com/problems/valid-parentheses/
        accum=[]
        for ss in s:
            if ss in "([{":
                accum.append(ss)
            else:
                if len(accum)==0:
                    return False
                if accum[-1]+ss in ["[]","{}","()"]:
                    accum.pop(-1)
                else:
                    return False
        if len(accum)==0:
            return True
        return False

    def evalRPN(self, tokens: List[str]) -> int:
        #https://leetcode.com/problems/evaluate-reverse-polish-notation/
        truncate = lambda a : int(a) if a>=0 else -1*int(-1*a)
        val_stack = []
        op1=lambda a,b: a+b
        op2 = lambda a,b: a-b
        op3= lambda a,b: a*b
        op4= lambda a,b:truncate(a/b)
        op_dict = {'+':op1, '-':op2, '*': op3, "/": op4}
        op_set = set(op_dict.keys())
        for t in tokens:
            if t in op_set:
                op = op_dict[t]
                b = val_stack.pop()
                a = val_stack.pop()
                v = op(a,b)
                val_stack.append(v)
            else:
                val_stack.append(int(t))
        return val_stack.pop()

    def generateParenthesis(self, n: int) -> List[str]:
        #https://leetcode.com/problems/generate-parentheses/
        p_dict=dict()
        for i in range(n+1):
            if i==0:
                p_dict[0]=[""]
            else:
                p_dict[i]=[]
                for j in range(i):
                    p_dict[i] += list(map(lambda a: "("+ a[0] + ")" + a[1], product(p_dict[j], p_dict[i-1-j])))


        return p_dict[n]

    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        #https://leetcode.com/problems/daily-temperatures/
        p_stack = []
        vals = [0 for _ in temperatures]

        for i, t in enumerate(temperatures):
            if len(p_stack) == 0:
                p_stack.append((t, i))

            else:
                while len(p_stack) > 0:
                    if p_stack[-1][0] >= t:
                        break
                    _, ind = p_stack.pop()
                    vals[ind] = i - ind
                p_stack.append((t, i))

        return vals

    def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:
        #https://leetcode.com/problems/car-fleet/
        cars = sorted(list(zip([target-p for p in position], speed)))
        f = 0
        p_f = None
        s_f = None
        for p,s in cars:
            if p_f is None:
                p_f = p
                s_f = s
                f+=1
            elif p/s > p_f/s_f:
                f+=1
                p_f = p
                s_f = s
        return f


    #https://leetcode.com/problems/largest-rectangle-in-histogram/
    def largestRectangleArea(self, heights: List[int]) -> int:
        st = []
        mx = 0
        # was approaching this solution through own efforts but missed part
        for i, h in enumerate(heights):
            ii = i
            while (len(st) > 0 and st[-1][0] >= h):
                hh, ii = st.pop()
                area = hh * (i - ii)
                mx = max(mx, area)
            st.append([h, ii])
            # this was the part I got kind of - we are keeping a stack
            # of things that have a rectangle forward
            # what I missed is giving it the last index of something less
            # so you get the rectangle left of it, too
        while (len(st) > 0):
            hh, ii = st.pop()
            mx = max(mx, hh * (len(heights) - ii))
        return mx

    def largestRectangleArea_incorrect2(self, heights: List[int]) -> int:
        # fails on 2,1,2

        m_box = heights[0]

        h_stack = [[heights[0], 0]]
        for i, h in enumerate(heights):
            if i == 0:
                continue

            m_box = max(m_box, h)
            while len(h_stack) > 0:
                if h < h_stack[-1][0]:
                    hh, ii = h_stack.pop()
                    m_box = max(m_box, (i - ii) * hh)
                else:
                    break

            h_stack.append([h, i])
        for h, i in h_stack:
            m_box = max(m_box, h * (h_stack[-1][1] - i + 1))
        return m_box

    def largestRectangleArea_try00(self, heights: List[int]) -> int:

        l_pntr = 0
        m_box = heights[0]
        for r_pntr in range(len(heights)):
            while heights[l_pntr] > heights[r_pntr]:
                m_box = max(m_box, heights[l_pntr] * (r_pntr - l_pntr))
                l_pntr += 1
        r_pntr = len(heights) - 1
        while l_pntr < len(heights):
            m_box = max(m_box, heights[l_pntr] * (r_pntr - l_pntr))
            l_pntr += 1
        return m_box

    def largestRectangleArea_incorrect(self, heights: List[int]) -> int:
        """
        too slow but I like what I coded here, going to keep for my git
        """
        val_l = sorted(list(set(heights.copy())))
        val_i = defaultdict(list)
        m_box = 0
        for i, h in enumerate(heights):
            val_i[h].append(i)
        i_s1 = [[0, len(heights) - 1]]
        i_s2 = []
        for h in val_l:
            while len(i_s1) > 0:
                iv = i_s1.pop()
                m_box = max(h * (iv[1] - iv[0] + 1), m_box)
                i_s2.append(iv)

            j = 0
            ind = val_i[h][0]
            while len(i_s2) > 0:
                iv = i_s2.pop()
                if ind is None:
                    i_s1.append(iv)
                    continue
                if ind > iv[0] and ind < iv[1]:
                    i_s1.append([iv[0], ind - 1])
                    i_s2.append([ind + 1, iv[1]])
                elif ind == iv[0] and ind == iv[1]:
                    pass
                elif ind == iv[0]:
                    i_s2.append([iv[0] + 1, iv[1]])
                elif ind == iv[1]:
                    i_s1.append([iv[0], iv[1] - 1])
                else:
                    i_s1.append(iv)
                    continue
                j += 1
                if j < len(val_i[h]):
                    ind = val_i[h][j]
                else:
                    ind = None
        return m_box

    #https://leetcode.com/problems/maximum-depth-of-binary-tree/
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if root is None:
            return 0
        max_d = 0
        node_stack = [(root, 1)]
        while node_stack:
            node_current, node_current_depth = node_stack.pop(0)
            max_d = max(node_current_depth, max_d)
            for n in [node_current.left, node_current.right]:
                if n is not None:
                    node_stack.append((n, node_current_depth + 1))

        return max_d

    #https://leetcode.com/problems/search-in-rotated-sorted-array/
    def search(self, nums: List[int], target: int) -> int:
        # first get rotation
        # search within rotation

        entry_b = nums[0]
        entry_e = nums[-1]
        l = 0
        r = len(nums)

        if len(nums) <= 3:
            min_n = min(nums)
            rotation = 0
            for i, n in enumerate(nums):
                if n == min_n:
                    min_n = n
                    rotation = i
        elif entry_b < entry_e:
            rotation = 0
        else:

            while True:
                mid = int((l + r) / 2)
                if nums[mid] > nums[mid + 1]:
                    rotation = mid + 1
                    break
                if nums[mid + 1] > entry_b:
                    l = mid
                elif nums[mid] < entry_e:
                    r = mid
        num_v = lambda i: nums[(i + rotation) % len(nums)]
        min_e = num_v(0)
        max_e = num_v(len(nums) - 1)
        l = 0
        r = len(nums) - 1
        if target > max_e or target < min_e:
            return -1
        while r - l >= 1:
            mid = int((l + r) / 2)
            if num_v(mid) == target:
                return (mid + rotation) % len(nums)
            if num_v(mid) > target:
                r = mid - 1
            else:
                l = mid + 1
        for i in [l, r]:
            if num_v(i) == target:
                return (i + rotation) % len(nums)
        return -1

    #https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/
    def findMin(self, nums: List[int]) -> int:
        # how does th
        entry_b = nums[0]
        entry_e = nums[-1]
        l = 0
        r = len(nums)

        if len(nums) <= 3:
            return min(nums)

        if entry_b < entry_e:
            return entry_b

        while True:
            mid = int((l + r) / 2)
            if nums[mid] > nums[mid + 1]:
                return nums[mid + 1]
            if nums[mid + 1] > entry_b:
                l = mid
            elif nums[mid] < entry_e:
                r = mid
    #https://leetcode.com/problems/koko-eating-bananas/
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        eating_time = lambda k : sum(map(lambda p: math.ceil(p/k), piles ))
        l = math.ceil(sum(piles)/h)
        r = max(piles)
        #new btree method
        # but note eating time decreases as speed goes up
        while (l<r):
            mid = int((l+r)/2)
            if eating_time(mid) <=h:
                r=mid
            else:
                l=mid+1
        return r

    def minEatingSpeed_incorrect(self, piles: List[int], h: int) -> int:
        min_speed=int(sum(piles)/h)
        max_speed=max(piles)
        try_speed = int((min_speed+max_speed)/2)


        eating_time = lambda k : sum(map(lambda p: math.ceil(p/k), piles ))
        m = max_speed
        while max_speed-min_speed>3:
            e=eating_time(try_speed)
            if e<=h:
                m = min(e,m)
            if eating_time(try_speed)<h:
                try_speed,max_speed = int((min_speed+try_speed)/2),try_speed
            elif eating_time(try_speed)>=h:
                try_speed,min_speed = int((max_speed+try_speed)/2),try_speed
        for i in range(min_speed, max_speed+1):
            e=eating_time(try_speed)
            if e<=h:
                m = min(e,m)
        return m


#STACK
class MinStack:
    #https://leetcode.com/problems/min-stack/
    def __init__(self):
        self.ind_stack = []
        self.val_stack = []
        self.n = 0
        self.m = None

    def push(self, val: int) -> None:
        self.val_stack.append(val)
        self.n += 1
        if self.m is None:
            self.ind_stack.append(0)
            self.m = val
        elif self.m > val:
            self.ind_stack.append(self.n - 1)
            self.m = val

    def pop(self) -> None:
        val = self.val_stack.pop()
        self.n -= 1
        if self.n > 0:
            if self.ind_stack[-1] == self.n:
                self.ind_stack.pop()
                self.m = self.val_stack[self.ind_stack[-1]]
        else:
            self.m = None
            self.ind_stack = []
        return val

    def top(self) -> int:
        return self.val_stack[-1]

    def getMin(self) -> int:
        return self.val_stack[self.ind_stack[-1]]












