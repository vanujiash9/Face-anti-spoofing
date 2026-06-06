import os
import csv
from collections import defaultdict, Counter

# ================= CONFIG ==================
LIVE_DIR = r"data_process\clean_live_sample"
NONLIVE_DIR = r"data_process\SpoofDataset"
OUTPUT_DIR = r"data_process\dataset_stats"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_ANNOTATION = os.path.join(OUTPUT_DIR, "table1_annotation.csv")
CSV_INDEX = os.path.join(OUTPUT_DIR, "table2_index.csv")
CSV_STATS = os.path.join(OUTPUT_DIR, "table3_stats.csv")



def get_files_with_labels(live_dir, nonlive_dir):
    all_files = []
    types_set = set()

    # --- Live ---
    for root, _, files in os.walk(live_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.relpath(os.path.join(root, f), start="data_process")
                all_files.append((path, 0, "live"))
    
    # --- Non-live ---
    for root, _, files in os.walk(nonlive_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
            
                name_parts = f.split("_")
                if len(name_parts) < 3:
                    type_label = "unknown"
                else:
                    type_label = "_".join(name_parts[2:]).split(".")[0]
                path = os.path.relpath(os.path.join(root, f), start="data_process")
                all_files.append((path, 1, type_label))
                types_set.add(type_label)

    return all_files, sorted(list(types_set))

all_files, spoof_types = get_files_with_labels(LIVE_DIR, NONLIVE_DIR)

# ----------------------------
# Bảng 1: Annotation chi tiết
# ----------------------------
with open(CSV_ANNOTATION, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    header = ["filepath"] + spoof_types + ["Live"]
    writer.writerow(header)

    for filepath, label, type_label in all_files:
        row = [filepath]
        for t in spoof_types:
            row.append("Yes" if t == type_label else "No")
        row.append("Yes" if label == 0 else "No")
        writer.writerow(row)

# ----------------------------
# Bảng 2: Dataset index
# ----------------------------
with open(CSV_INDEX, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filepath", "label", "type"])
    for filepath, label, type_label in all_files:
        writer.writerow([filepath, label, type_label])

# ----------------------------
# Bảng 3: Thống kê tổng hợp
# ----------------------------
counter_types = Counter()
live_count = 0
for _, label, type_label in all_files:
    if label == 0:
        live_count += 1
    else:
        counter_types[type_label] += 1

total_nonlive = sum(counter_types.values())
total_files = total_nonlive + live_count
top5_nonlive = counter_types.most_common(5)

with open(CSV_STATS, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Category", "Count"])
    for t in spoof_types:
        writer.writerow([t, counter_types.get(t, 0)])
    writer.writerow(["Total Non-live", total_nonlive])
    writer.writerow(["Live", live_count])
    writer.writerow(["Total files", total_files])
    writer.writerow(["Live/Non-live ratio", f"{live_count}:{total_nonlive}"])
    writer.writerow([])
    writer.writerow(["Top 5 Non-live types", "Count"])
    for t, c in top5_nonlive:
        writer.writerow([t, c])

print(" 3 CSV đã được tạo:")
print(f" - {CSV_ANNOTATION}")
print(f" - {CSV_INDEX}")
print(f" - {CSV_STATS}")
