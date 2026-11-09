#!/usr/bin/env python3
"""BD-rate (percent) of test curve vs anchor curve. Usage: bd.py "r1,p1 r2,p2 ..." "r1,p1 ..." """
import sys, math
def pts(s): return sorted([tuple(map(float,x.split(','))) for x in s.split()], key=lambda t:t[1])
def fit(p):
    # cubic fit of log(rate) as a function of psnr via least squares (Vandermonde, small n)
    import itertools
    xs=[q for _,q in p]; ys=[math.log(r) for r,_ in p]; n=min(3,len(p)-1)
    A=[[x**k for k in range(n+1)] for x in xs]
    # normal equations
    M=[[sum(A[i][a]*A[i][b] for i in range(len(xs))) for b in range(n+1)] for a in range(n+1)]
    v=[sum(A[i][a]*ys[i] for i in range(len(xs))) for a in range(n+1)]
    for c in range(n+1):
        piv=max(range(c,n+1),key=lambda r:abs(M[r][c])); M[c],M[piv]=M[piv],M[c]; v[c],v[piv]=v[piv],v[c]
        for r in range(n+1):
            if r!=c:
                f=M[r][c]/M[c][c]
                M[r]=[M[r][k]-f*M[c][k] for k in range(n+1)]; v[r]-=f*v[c]
    return [v[k]/M[k][k] for k in range(n+1)]
def integ(c,a,b): return sum(c[k]*(b**(k+1)-a**(k+1))/(k+1) for k in range(len(c)))
a=pts(sys.argv[1]); t=pts(sys.argv[2])
lo=max(a[0][1],t[0][1]); hi=min(a[-1][1],t[-1][1])
if hi<=lo: print("no overlap"); sys.exit()
ca=fit(a); ct=fit(t)
d=(integ(ct,lo,hi)-integ(ca,lo,hi))/(hi-lo)
print(f"BD-rate {100*(math.exp(d)-1):+.1f}% (psnr {lo:.1f}..{hi:.1f})")
