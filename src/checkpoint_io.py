import os
import shutil
import time


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def atomic_copy_with_backup(src: str, dst: str, make_backup: bool = True):
    """
    Copy src to dst through a temporary file in dst's directory, then atomically
    replace dst. Existing dst is copied to a timestamped backup before replace.
    """
    src_path = os.path.abspath(src)
    dst_path = os.path.abspath(dst)

    if src_path == dst_path:
        print(f"write-back skipped: source and target are same path: {dst}")
        return {"dst": dst, "backup": None, "skipped": True}

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"source model not found: {src}")

    ensure_parent(dst_path)
    dst_dir = os.path.dirname(dst_path) or "."
    dst_name = os.path.basename(dst_path)
    tmp_path = os.path.join(dst_dir, f".{dst_name}.tmp_{os.getpid()}_{timestamp()}")

    backup_path = None
    try:
        shutil.copy2(src_path, tmp_path)

        if make_backup and os.path.exists(dst_path):
            backup_path = f"{dst_path}.bak_{timestamp()}"
            shutil.copy2(dst_path, backup_path)
            print(f"backup old model: {backup_path}")

        os.replace(tmp_path, dst_path)
        print(f"atomic write-back: {src} -> {dst}")
        return {"dst": dst, "backup": backup_path, "skipped": False}
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
