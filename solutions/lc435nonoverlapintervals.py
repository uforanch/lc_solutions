from typing import List

class Solution:
    def binsearch(self, lst, tar):
        # need index of biggest item
        # smaller than tar
        if not lst:
            return -1
        if len(lst) == 1:
            return 0 if lst[0] <= tar else -1
        if lst[len(lst) - 1] <= tar:
            return len(list - 1)
        if lst[0] > tar:
            return -1
        m = (len(lst)) // 2
        while True:
            if lst[m] <= tar:
                if lst[m + 1] > tar:
                    return m
                m = int((m + len(lst)) / 2)
            if lst[m] > tar:
                if lst[m - 1] <= tar:
                    return m - 1
                m = m // 2

    def eraseOverlapIntervals_unopt(self, intervals: List[List[int]]) -> int:
        """
        dynamic programming
        work intervals from left to right (by right end)

        recurrance is take next right interval (s e), see which right ends are in there
        include interval - don't include any interval with those right ends (DP( E <s))
        don't include interval, use last end
        more than one intercal at the end have to try each one

        here's what we'll do - sort the ends and also make hash map

        could probably just sort by end then start actually... but need to know how many ends there are
        """
        intervals.sort(key=lambda x: (x[1], x[0]))
        end_arr = []
        start_pointer = -1
        dp = [0]
        """
        value of dp - max intervals with ends at end_arr[s_pointer+1]
        """
        for interval in intervals:
            if not end_arr:
                end_arr.append(interval[1])
                dp.append(0)
            elif end_arr[-1] != interval[1]:
                end_arr.append(interval[1])
                start_pointer = self.binsearch(end_arr, interval[0])
                dp.append(dp[-1])
            else:
                while end_arr[start_pointer + 1] <= interval[0]:
                    start_pointer += 1

            dp[-1] = max(dp[start_pointer + 1] + 1, dp[-1])
        return len(intervals)-dp[-1]

    def eraseOverlapIntervals_dl(self, intervals: List[List[int]]) -> int:
        res = 0

        intervals.sort(key=lambda x: x[1])
        prev_end = intervals[0][1]

        for i in range(1, len(intervals)):
            if prev_end > intervals[i][0]:
                res += 1
            else:
                prev_end = intervals[i][1]

        return res
s = Solution()
i = [[1,2],[2,3],[3,4],[1,3]]

#1
i=[[1,2],[1,2],[1,2]]
#2?
i=[[1,2],[2,3]]
#0
i=[[1,3],[2,4],[3,5]]
print(s.eraseOverlapIntervals_dl(i))