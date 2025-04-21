class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        arr1c = [str(x) for x in arr1]
        arr2c = [str(x) for x in arr2]
        arr1c.sort(key=lambda x: len(x) * -1)
        arr2c.sort(key=lambda x: len(x) * -1)
        longest_pref = 0
        for a1 in arr1c:
            for a2 in arr2c:
                if len(a1) < longest_pref or len(a2) < longest_pref:
                    continue
                while a1[:longest_pref] == a2[:longest_pref]:
                    longest_pref += 1
                    if len(a1) < longest_pref or len(a2) < longest_pref:
                        break
        if longest_pref == 0:
            return 0
        return longest_pref - 1
