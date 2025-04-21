class Solution:
    """
    used a different description of solution

    lessons learned:
    Make sure cases you are iterating through include the final case
    Have clarity of what cases are taken

    2d also worked a lot faster than subset

    """
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        if len(s1) + len(s2) != len(s3):
            return False
        if len(s3) == 0:
            return True
        if len(s2) == 0 or len(s1) == 0:
            return s1 + s2 == s3
        if len(s1) > len(s2):
            self.isInterleave(s2, s1, s3)

        dp = [[False for _ in range(len(s2)+1)] for _ in range(len(s1)+1)]
        dp[0][0] = True
        for i in range(1, len(s1)+1):
            dp[i][0] = (s3[i - 1] == s1[i - 1]) and dp[i - 1][0]
        for i in range(1, len(s2)+1):
            dp[0][i] = (s3[i - 1] == s2[i - 1]) and dp[0][i - 1]

        # this strictly DOES NOT need to be done
        # doing it for practice
        for k in range(2, len(s3)+1):
            for r in range(max(1, k - len(s2)), min(k, len(s1)+1)):
                c = k - r
                #print(r, c)
                #print(s1[:r], s2[:c], s3[:k])


                dp[r][c] = (dp[r - 1][c] and s1[r - 1] == s3[k - 1]) or (dp[r][c - 1] and s2[c - 1] == s3[k - 1])
                #print(dp[r][c])
        return dp[len(s1) ][len(s2) ]

    def isInterleaveRec(self, s1: str, s2: str, s3: str) -> bool:
        if len(s1) + len(s2) != len(s3):
            return False
        if len(s1) == 0 or len(s2) == 0:
            return s3 == s1 + s2
        if len(s3) == 0:
            return True
        if s1[0] != s3[0] and s2[0] != s3[0]:
            return False
        if s1[0] == s2[0]:
            return self.isInterleaveRec(s1[1:], s2, s3[1:]) or self.isInterleaveRec(s1, s2[1:], s3[1:])
        if s1[0] == s3[0]:
            return self.isInterleaveRec(s1[1:], s2, s3[1:])
        elif s2[0] == s3[0]:
            return self.isInterleaveRec(s1, s2[1:], s3[1:])
s = Solution()
print(s.isInterleave("aabcc","dbbca","aadbbcbcac")) #true
print(s.isInterleave("aabcc","dbbca","aadbbbaccc")) #false
print(s.isInterleave("aa","ab","abaa"))
