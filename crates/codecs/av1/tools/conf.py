#!/usr/bin/env python3
"""Generate AV1 streams with ffmpeg encoders and compare our decoder to libdav1d."""
import subprocess, sys, os, hashlib
D=os.environ.get('AV1TEST', os.path.expanduser('~/av1test'))
DEC=os.environ.get('AV1DEC', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../../target/release/examples/av1dec'))
os.makedirs(D,exist_ok=True)
def run(cmd):
    return subprocess.run(cmd,shell=True,capture_output=True,text=True)
def make(name, src, pix, enc, frames=3, size='352x288', extra_src=''):
    ivf=f'{D}/{name}.ivf'
    if not os.path.exists(ivf):
        r=run(f"ffmpeg -hide_banner -loglevel error -f lavfi -i {src}=size={size}:rate=30{extra_src} -frames:v {frames} -pix_fmt {pix} {enc} {ivf} -y")
        if r.returncode: print(name,'ENCODE FAIL',r.stderr[-300:]); return None
    return ivf
def ref(ivf, pix):
    out=ivf[:-4]+'.ref'
    if not os.path.exists(out):
        r=run(f"ffmpeg -hide_banner -loglevel error -c:v libdav1d -filmgrain 1 -i {ivf} -autoscale 0 -f rawvideo -pix_fmt {pix} {out} -y")
        if r.returncode: print('REF FAIL',r.stderr[-300:]); return None
    return out
def check(name, ivf, pix):
    rf=ref(ivf,pix)
    if not rf: return False
    r=run(f"{DEC} {ivf} {D}/{name}.out")
    if r.returncode:
        print(f'{name:40s} DECODE ERROR {r.stderr.strip()[-200:]}'); return False
    a=open(rf,'rb').read(); b=open(f'{D}/{name}.out','rb').read()
    if a==b:
        print(f'{name:40s} OK   ({r.stderr.strip()})'); return True
    # find first mismatch
    n=min(len(a),len(b)); i=next((k for k in range(n) if a[k]!=b[k]), n)
    print(f'{name:40s} MISMATCH len {len(a)} vs {len(b)} first diff at byte {i}')
    return False
CASES=[]
def case(name, pix, enc, src='testsrc2', frames=3, size='352x288', outpix=None, extra_src=''):
    CASES.append((name,pix,enc,src,frames,size,outpix or pix,extra_src))
if __name__=='__main__':
    import cases
    cases.define(case)
    sel=sys.argv[1:] 
    ok=0;tot=0
    for (name,pix,enc,src,frames,size,outpix,es) in CASES:
        if sel and not any(s in name for s in sel): continue
        ivf=make(name,src,pix,enc,frames,size,es)
        if not ivf: continue
        tot+=1; ok+=check(name,ivf,outpix)
    print(f'{ok}/{tot} passed')
