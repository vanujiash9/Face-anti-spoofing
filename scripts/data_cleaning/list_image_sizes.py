#Tìm ảnh có kích cỡ phù hợp
import argparse
import json
import os
from collections import defaultdict
from PIL import Image

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}


def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in IMG_EXTS


import concurrent.futures
from functools import partial

def _get_image_size(args):
    rel, fp = args
    try:
        with Image.open(fp) as im:
            w, h = im.size
        return (rel, f"{w}x{h}", None)
    except Exception as e:
        return (rel, None, str(e))

def collect_sizes(root_dir, follow_symlinks=False, num_workers=None):
    # 1. Thu thập danh sách file ảnh
    file_list = []  # (rel_folder, full_path)
    for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=follow_symlinks):
        rel = os.path.relpath(dirpath, root_dir)
        if rel == '.':
            rel = ''
        for f in filenames:
            if is_image_file(f):
                file_list.append((rel, os.path.join(dirpath, f)))
    # 2. Đọc song song
    results = defaultdict(lambda: defaultdict(int))
    errors = 0
    total = len(file_list)
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for rel, size, err in executor.map(_get_image_size, file_list, chunksize=32):
            if size:
                results[rel][size] += 1
            else:
                errors += 1
    return results, total, errors


def summary_to_jsonable(results):
    # Convert nested defaultdicts to plain dicts
    return {folder: dict(sizes) for folder, sizes in results.items()}


def print_summary(results, total, errors, top_n=10):
    print(f"Total images scanned: {total}")
    print(f"Errors reading images: {errors}")
    # Aggregate across all folders
    agg = defaultdict(int)
    for folder, sizes in results.items():
        for size, cnt in sizes.items():
            agg[size] += cnt
    print("\nTop sizes across dataset:")
    for size, cnt in sorted(agg.items(), key=lambda x: -x[1])[:top_n]:
        print(f"  {size}: {cnt}")


def write_output(out_path, results):
    payload = {
        'folders': summary_to_jsonable(results)
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='Collect image sizes per folder in a dataset')
    parser.add_argument('dataset_dir', help='Path to dataset folder to scan')
    parser.add_argument('--output', '-o', default='image_sizes_summary.json', help='Output JSON file')
    parser.add_argument('--follow-symlinks', action='store_true', help='Follow symlinks when walking')
    parser.add_argument('--workers', '-j', type=int, default=None, help='Number of parallel workers (default: CPU count)')
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_dir):
        print(f"Error: dataset directory not found: {args.dataset_dir}")
        raise SystemExit(2)

    results, total, errors = collect_sizes(args.dataset_dir, follow_symlinks=args.follow_symlinks, num_workers=args.workers)
    write_output(args.output, results)
    print_summary(results, total, errors)
    print(f"Wrote summary to {args.output}")


if __name__ == '__main__':
    main()
