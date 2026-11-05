#!/bin/bash
# Decode every test stream and compare with the libdav1d reference.
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${1:-$HERE/../../../../target/release/examples/av1dec}
AV1TEST=${AV1TEST:-$HOME/av1test}
ok=0; bad=0
for f in $AV1TEST/*.ivf; do r=${f%.ivf}.ref; [ -f $r ] || continue
  if $BIN $f ${TMPDIR:-/tmp}/av1chk.yuv 2>/dev/null && cmp -s ${TMPDIR:-/tmp}/av1chk.yuv $r; then ok=$((ok+1)); else bad=$((bad+1)); echo "FAIL $f"; fi
done
echo "$ok ok, $bad failed"
