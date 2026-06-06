import os
import cv2
import csv
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import heapq


def is_video_file(fname):
    ext = os.path.splitext(fname)[1].lower()
    return ext in {'.mp4', '.mov', '.avi', '.mkv', '.wmv'}


def sharpness(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def process_video(video_path, input_root, output_root):
    rows = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return rows

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            total = 0

        sample_limit = 300
        step = 1
        if total > sample_limit:
            step = max(1, total // sample_limit)

        best = []
        idx = 0
        read_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                var = sharpness(gray)
                if len(best) < 3:
                    heapq.heappush(best, (var, idx, frame.copy()))
                else:
                    if var > best[0][0]:
                        heapq.heapreplace(best, (var, idx, frame.copy()))
                read_idx += 1
            idx += 1

        cap.release()

        if not best:
            print(f"No frames sampled for {video_path}")
            return rows

        best_sorted = sorted(best, key=lambda x: -x[0])

        rel = os.path.relpath(video_path, input_root)
        rel_dir = os.path.dirname(rel)
        video_base = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(output_root, rel_dir)
        os.makedirs(out_dir, exist_ok=True)

        saved_files = []
        for i, (var, frame_idx, img) in enumerate(best_sorted, start=1):
            out_name = f"{video_base}__f{frame_idx:06d}__r{i}.jpg"
            out_path = os.path.join(out_dir, out_name)
            cv2.imwrite(out_path, img)
            saved_files.append(out_path)

        parts = rel.split(os.sep)
        type_name = parts[0] if len(parts) >= 1 else ''
        label = parts[1] if len(parts) >= 2 else video_base

        for sf in saved_files:
            norm = os.path.normpath(sf)
            rows.append({'path': norm.replace('\\', '/'), 'label': label, 'type': type_name})

        print(f"Processed video: {video_path}")
        for s in saved_files:
            print(f"  -> {s}")
        return rows

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return rows


def find_videos(root):
    videos = []
    for dirpath, dirs, files in os.walk(root):
        for f in files:
            if is_video_file(f):
                videos.append(os.path.join(dirpath, f))
    return videos


def main():
    parser = argparse.ArgumentParser(description='Extract top-3 sharp frames from videos under a folder')
    parser.add_argument('--input', '-i', default=os.path.join('dataset', 'Slicon'), help='Input folder containing videos')
    parser.add_argument('--output', '-o', default=os.path.join('clean_live_sample', 'Slicon_frames'), help='Output folder for extracted frames')
    parser.add_argument('--csv', '-c', default=os.path.join('attack_tables_full', 'slicon_frames.csv'), help='CSV output file')
    parser.add_argument('--workers', '-w', type=int, default=4, help='Number of worker threads')
    args = parser.parse_args()

    input_root = args.input
    output_root = args.output
    csv_path = args.csv

    videos = find_videos(input_root)
    if not videos:
        print(f"No videos found under {input_root}")
        return

    all_rows = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_video, v, input_root, output_root): v for v in videos}
        for fut in as_completed(futures):
            rows = fut.result()
            all_rows.extend(rows)

    # write CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'label', 'type'])
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    print(f"Wrote CSV: {csv_path} with {len(all_rows)} rows")


if __name__ == '__main__':
    main()
