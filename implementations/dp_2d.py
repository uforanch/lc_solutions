
def dp_case_length_iteration(n,m):
    for r in range(n+1):
        line_str = ""
        for c in range(m+1):
            line_str += f" {r+c} ({r}, {c})"
        print(line_str)
        #NOTE: could just make a hash from r+c to coords
        #but let's be mathy
    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
    #case usually includes n,m so be inclusive
    #base case will do the top edge and bottom edge
    #want 1<=r<=n, 1<=c<=m
    #therefore k-1>=c>=k-m
    for k in range(2, n+m+1):
        print(k)
        for r in range(max(1,k-m), min(n+1, k)):
            c=k-r
            print("(", r,c ,")")
dp_case_length_iteration(3,4)