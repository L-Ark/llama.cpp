# GP40 remote storage audit for full IQ1_S smoke

Timestamp: `2026-07-07T06:10:50+0800`.

Branch:
`vendor/kimi-speculative-general-token-rate-16gb`.

Commit audited on remote:
`8257e4afccd6ce12d9c5b5d8e566358c51b6d5a4`.

Remote worktree:
`/root/lfz/tmp/vendor-kimi-speculative-gp33`.

Raw output:
`.Agent/runs/20260707-gp40-remote-storage-audit/raw.txt`.

Command:

```bash
ssh -p 51056 root@92.180.27.82 'set -euo pipefail
cd /root/lfz/tmp/vendor-kimi-speculative-gp33
git fetch origin vendor/kimi-speculative-general-token-rate-16gb >/dev/null
git reset --hard origin/vendor/kimi-speculative-general-token-rate-16gb >/dev/null
printf "repo_head="; git rev-parse HEAD
printf "date="; date -Is
printf "required_iq1s_bytes=204430872480\n"
printf "required_with_20gib_reserve=225905708960\n"
df -B1 -T
findmnt -b -o TARGET,SOURCE,FSTYPE,SIZE,AVAIL,OPTIONS
lsblk -b -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS || true
for p in /root /root/lfz /root/lfz/models /root/lfz/runs /mnt /data /workspace /tmp; do
  if [ -e "$p" ]; then
    printf "path=%s exists=1 " "$p"
    df -B1 --output=target,fstype,size,avail "$p" | tail -n 1
  else
    printf "path=%s exists=0\n" "$p"
  fi
done
du -x -B1 -d1 /root/lfz 2>/dev/null | sort -n || true
du -x -B1 -d1 /root/lfz/models 2>/dev/null | sort -n || true
du -x -B1 -d2 /root/lfz/runs 2>/dev/null | sort -n | tail -n 40 || true'
```

Required bytes:

- IQ1_S final GGUF: `204430872480`.
- IQ1_S plus 20 GiB reserve: `225905708960`.

Filesystem finding:

- Only one writable persistent filesystem is available for the relevant paths:
  `/dev/vda1` mounted at `/`, ext4.
- Available bytes on `/`: `89639956480`.
- `/root`, `/root/lfz`, `/root/lfz/models`, `/root/lfz/runs`, `/mnt`, and
  `/tmp` all resolve to the same `/` filesystem with the same free space.
- `/data` does not exist.
- `/workspace` does not exist.
- `/dev/shm` and `/run/qemu` each have `35369054208` bytes available, which is
  far below the IQ1_S requirement and is tmpfs, not a persistent model location.

Disk consumers:

- `/root/lfz/models`: `405392715776` bytes.
- `/root/lfz/runs`: `469723262976` bytes.
- `/root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S`: `405392306176` bytes.
- `/root/lfz/runs/ik_llama/kimi-iq3s-assets`: `444499689472` bytes.

Decision:

- There is no alternate mounted filesystem that can hold full IQ1_S with the
  required 20 GiB reserve.
- Overriding `MODEL_DIR` to `/mnt`, `/tmp`, or another existing path cannot
  bypass the disk gate because those paths are on the same root filesystem.
- Full IQ1_S n32 smoke still requires one of:
  - explicit approval to delete old non-SOTA Kimi expert packs;
  - external storage / larger disk;
  - a different byte-reduction path that does not require storing the full
    IQ1_S GGUF.
- No deletion, download, move, truncate, or smoke was performed.
- No SOTA or token-rate claim is made from this phase.
