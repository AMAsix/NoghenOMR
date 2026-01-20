"""
Post-install script to patch homr for numpy 2.x compatibility.

Run this script after `pip install -r requirements.txt` to fix
"only 0-dimensional arrays can be converted to Python scalars" errors.

Usage:
    python patches/apply_homr_numpy2_patches.py
"""

import os
import sys
import re
from pathlib import Path


def find_site_packages():
    """Find the site-packages directory containing homr."""
    for path in sys.path:
        homr_path = Path(path) / "homr"
        if homr_path.exists() and homr_path.is_dir():
            return Path(path)
    return None


def patch_file(filepath: Path, patterns: list[tuple[str, str]]):
    """Apply regex replacements to a file."""
    if not filepath.exists():
        print(f"  WARNING: {filepath} not found, skipping")
        return False
    
    content = filepath.read_text(encoding='utf-8')
    original_content = content
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    if content != original_content:
        filepath.write_text(content, encoding='utf-8')
        print(f"  Patched: {filepath}")
        return True
    else:
        print(f"  Already patched or no match: {filepath}")
        return False


def main():
    print("Applying homr numpy 2.x compatibility patches...")
    
    site_packages = find_site_packages()
    if site_packages is None:
        print("ERROR: Could not find homr installation in site-packages")
        sys.exit(1)
    
    homr_path = site_packages / "homr"
    print(f"Found homr at: {homr_path}")
    
    patches = [
        # autocrop.py - fix int(x[1]) in lambda for histogram max
        (
            homr_path / "autocrop.py",
            [
                (
                    r'key=lambda x: int\(x\[1\]\)',
                    'key=lambda x: int(np.asarray(x[1]).item())'
                ),
            ]
        ),
        # color_adjust.py - fix int(center_of_mass) and int(np.max(...))
        (
            homr_path / "color_adjust.py",
            [
                (
                    r'return int\(center_of_mass\)',
                    'return int(np.asarray(center_of_mass).item())'
                ),
                (
                    r'max_background = int\(np\.max\(background_blurred\[valid_background\]\)\)',
                    'max_background = int(np.asarray(np.max(background_blurred[valid_background])).item())'
                ),
            ]
        ),
        # bounding_boxes.py - fix int(self.box[N]) calls
        (
            homr_path / "bounding_boxes.py",
            [
                (
                    r'\(int\(self\.box\[0\]\), int\(self\.box\[1\]\)\)',
                    '(int(np.asarray(self.box[0]).item()), int(np.asarray(self.box[1]).item()))'
                ),
                (
                    r'\(int\(self\.box\[2\]\), int\(self\.box\[3\]\)\)',
                    '(int(np.asarray(self.box[2]).item()), int(np.asarray(self.box[3]).item()))'
                ),
            ]
        ),
    ]
    
    patched_count = 0
    for filepath, file_patterns in patches:
        if patch_file(filepath, file_patterns):
            patched_count += 1
    
    print(f"\nPatching complete. {patched_count} file(s) modified.")
    print("Note: Re-run this script after reinstalling homr.")


if __name__ == "__main__":
    main()
