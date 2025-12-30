import os
import cv2
import csv
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

SRC_DIR = r"dataset\iBeta Level 2"
DST_DIR = "IBeta"
CSV_INDEX = os.path.join(DST_DIR, "IBeta_index.csv")
CSV_MAP = os.path.join(DST_DIR, "IBeta_original_map.csv")
TOTAL_TARGET = 10000
SPOOF_TYPES = {"latex", "silicon", "wraped3d"}

os.makedirs(DST_DIR, exist_ok=True)

def get_best_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        return None

    best_score = -1
    best_frame = None
    step = max(frame_count // 30, 1)
    for i in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if score > best_score:
            best_score = score
            best_frame = frame
    cap.release()
    return best_frame

def process_video(args):
    video_path, spoof_type, idx = args
    frame = get_best_frame(video_path)
    if frame is None:
        return None
    new_name = f"IBeta{idx:06d}_1_{spoof_type}.jpg"
    save_path = os.path.join(DST_DIR, new_name)
    cv2.imwrite(save_path, frame)
    return (new_name, 1, spoof_type, video_path)

def gather_videos(src_dir):
    videos = []
    for root, dirs, files in os.walk(src_dir):
        for f in files:
            if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                rel_path = os.path.relpath(root, SRC_DIR)
                spoof_type = rel_path.split(os.sep)[0].lower().replace("_", "")
                if spoof_type not in SPOOF_TYPES:
                    continue
                videos.append((os.path.join(root, f), spoof_type))
    return videos

def main():
    start_time = datetime.now()
    videos = gather_videos(SRC_DIR)
    videos = videos[:TOTAL_TARGET]

    args_list = [(vp, spf, idx+1) for idx, (vp, spf) in enumerate(videos)]

    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_video, args) for args in args_list]
        for f in as_completed(futures):
            res = f.result()
            if res:
                results.append(res)

    with open(CSV_INDEX, "w", newline="", encoding="utf-8") as f_idx, \
         open(CSV_MAP, "w", newline="", encoding="utf-8") as f_map:
        writer_idx = csv.writer(f_idx)
        writer_map = csv.writer(f_map)
        writer_idx.writerow(["path", "label", "type"])
        writer_map.writerow(["original_path", "new_name"])
        for new_name, label, spoof, orig_path in results:
            writer_idx.writerow([new_name, label, spoof])
            writer_map.writerow([orig_path, new_name])

    print("------ XỬ LÝ HOÀN THÀNH ------")
    print(f"Tổng số video đã xử lý: {len(results)}")
    print(f"CSV Index: {CSV_INDEX}")
    print(f"CSV Original Map: {CSV_MAP}")
    print(f"Tổng thời gian: {datetime.now() - start_time}")

if __name__ == "__main__":
    main()
