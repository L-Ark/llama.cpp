# GP107 remote storage gate for lower-quant smoke

Date: 2026-07-08

This is a non-destructive inventory. No files were deleted or moved.

## Purpose

Check whether the remote server has an alternate mount point or external
storage location that can hold the `i1-IQ1_S` lower-quant GGUF candidate while
preserving the existing IQ3_S SOTA reproduction assets and the required
`50 GiB` safety reserve.

## Result

No alternate storage was found.

- The server exposes only one usable large filesystem for the relevant paths:
  `/dev/root` mounted at `/`.
- `/mnt`, `/home`, `/tmp`, `/var/tmp`, `/root`, and `/root/lfz` all resolve to
  the same `/dev/root` filesystem.
- Current available bytes on `/dev/root`: `243713118208`.
- Required bytes for `i1-IQ1_S` plus the existing `50 GiB` reserve:
  `258117963680`.
- Current shortfall: about `14.4 GB`.

The previous deletion candidate still exists:

```text
17179111424 /root/lfz/runs/ik_llama/kimi-iq3s-assets/kimi-iq3s-general-dev-budget16-overlay.expert-pack
```

Deleting that old non-current overlay would satisfy the disk gate, but no
deletion has been performed because it requires explicit user approval.

## Evidence

Filesystem summary:

```text
Filesystem Type 1B-blocks Used Available Use% Mounted on
/dev/root ext4 1065418129408 821688233984 243713118208 78% /
```

Relevant block device summary:

```text
vda disk 1099511627776
|-vda1 part 1099395218944 ext4 /
|-vda14 part 4194304
`-vda15 part 111149056 vfat /boot/efi
```

Top-level `/root/lfz` usage:

```text
735753080832 /root/lfz
405839765504 /root/lfz/models
405392306176 /root/lfz/models/Kimi-K2.7-Code-GGUF-IQ3_S
316592959488 /root/lfz/runs
303546454016 /root/lfz/runs/ik_llama
11444740096  /root/lfz/runs/vendor-kimi-token-rate
4519452672   /root/lfz/tmp
```

## Reproduce

```bash
ssh -p 51056 root@92.180.27.82 'set -e
echo "## df"
df -B1 -T
echo "## lsblk"
lsblk -b -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS,MODEL 2>/dev/null || true
echo "## top-level du /root/lfz"
du -x -B1 -d 2 /root/lfz 2>/dev/null | sort -nr | head -80
echo "## common dirs"
for d in /mnt /data /workspace /home /tmp /var/tmp /root /root/lfz; do
  [ -e "$d" ] && printf "%s\n" "$d" && df -B1 "$d"
done'
```

## Decision

The lower-quant smoke cannot proceed non-destructively on the current server
without either:

- explicit approval to delete at least the old
  `kimi-iq3s-general-dev-budget16-overlay.expert-pack`; or
- a new external storage location with enough free space.

