#https://www.geeksforgeeks.org/minimum-word-break/

#trie from memory

class TrieNode:
    def __init__(self, val=""):
        self.val = val
        self.children = dict()
    def insert(self, s: str):
        if not s.startswith(self.val):
            return
        s = s[len(self.val):]
        if len(s)==0:
            return
        if s[0] in self.children.keys():
            t = self.children[s[0]]
        else:
            t = TrieNode(s[0])
            self.children[s[0]] = t
        t.insert(s)
    def contains(self, s):
        if not s.startswith(self.val):
            return False
        s=s[len(self.val)]
        if len(s)==0:
            return True
        if s[0] not in self.children.keys():
            return False
        return self.children[s[0]].contains(s)
    def follow(self, s):
        if not s.startswith(self.val):
            return ""
        s=s[len(self.val):]
        if len(s)==0:
            return self.val
        if s[0] not in self.children.keys():
            return self.val
        return self.val + self.children[s[0]].follow(s)
    def count_all(self):
        if not self.children:
            return 1
        s=0
        for c in self.children:
            s+= self.children[c].count_all()
        return s+1
def solve(wordlist, word):
    t = TrieNode()
    for w in wordlist:
        t.insert(w)
    outlist = []
    while len(word)>0:
        f = t.follow(word)
        if (len(f)==0):
            break
        outlist.append(f)
        word = word[len(f):]
    print(outlist)
    return outlist


def min_word_break_dl(s, wordlist):
    n = len(s)
    dp = [float('inf')] * (n + 1)  # Initialize a list to store minimum word breaks needed
    dp[0] = 0  # No breaks needed for an empty string

    for i in range(1, n + 1):
        for word in wordlist:
            length = len(word)
            if i >= length and s[i - length:i] == word:
                # Check if the substring from i-length to i matches a word in the wordlistionary
                dp[i] = min(dp[i], dp[i - length] + 1)
                # Update the minimum word breaks needed at position i

    return dp[n] - 1  # Return the minimum word breaks needed for the entire string

def main(s):
    wordlist = ["Cat", "Mat", "Ca", "Ma", "at", "C", "Dog", "og", "Do"]
    min_breaks = min_word_break_dl(s, wordlist)
    print(f"The minimum word breaks needed: {min_breaks}")

solve(["Cat", "Mat", "Ca",
     "tM", "at", "C", "Dog", "og", "Do"], "CatMat")

solve(["Cat", "Mat", "Ca",
     "tM", "at", "C", "Dog", "og", "Do"], "Dogcat")

main("CatMatat")
main("DogCat")
main("Dogcat")

def count_substrings(word):
    t=TrieNode()
    for i in range(len(word)):
        t.insert(word[i:])
    c = t.count_all()
    return c

count_substrings("ab")
print("----")
count_substrings("ababab")