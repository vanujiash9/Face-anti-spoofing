import argparse
import csv
import os
from collections import Counter
from pathlib import Path
#Tạo 3 file csv 
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.rmvb', '.mpeg', '.mpg'}

DEFAULT_LIVE_FOLDERS = {'FFHQ', 'VGG'}

ATTACK_KEYWORDS = {
    'Printed': ['print', 'printed', 'print_'],
    'Mask Paper': ['paper', 'mask', 'paper_mask', 'mask_paper'],
    'SiliconeMask': ['silicone', 'silicon', 'silicone mask', 'silicone_mask'],
    'On Screen': ['phone', 'screen', 'poster', 'display', 'screen_', 'phone_','phone'],
    'Poster': ['poster'],
    'PhotoReplay': ['photo', 'replay', 'photo_'],
    'PadMask': ['pad', 'pad_mask'],
    'UpperBodyMask': ['upperbody', 'upper_body', 'upperbodymask', 'upper_body_mask'],
    'RegionMask': ['region', 'region_mask'],
    '3DMask': ['3d', '3d_mask', '3dmask'],
    'Deepfake': ['deepfake', 'deepfakes'],
    'Face2Face': ['face2face', 'face2face_'],
    'Swap/Defect (Injection)': ['faceswap', 'face_swap', 'faceshifter', 'face_shifter'],
    'NeuralTextures': ['neuraltextures', 'neural_textures']
}

def is_media_file(fname):
    ext = os.path.splitext(fname)[1].lower()
    return ext in IMAGE_EXTS or ext in VIDEO_EXTS

def file_type(fname):
    ext = os.path.splitext(fname)[1].lower()
    if ext in IMAGE_EXTS: return 'image'
    if ext in VIDEO_EXTS: return 'video'
    return 'other'

def classify_attack(path_parts, fname_lower):
    text = '/'.join(part.lower() for part in path_parts) + '/' + fname_lower
    matches = set()
    for attack, keys in ATTACK_KEYWORDS.items():
        for k in keys:
            if k in text:
                matches.add(attack)
                break
    return matches

def infer_attack_types_from_parts(parts, live_folders):
    if not parts: return (0, set(['Unspecified_NonLive']))
    top = parts[0].lower()
    if top in (p.lower() for p in live_folders): return (1, set())
    return (0, {parts[0]})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir')
    parser.add_argument('--output', '-o', default='attack_tables')
    parser.add_argument('--live-folders', nargs='*', default=list(DEFAULT_LIVE_FOLDERS))
    parser.add_argument('--min-sample-report', type=int, default=0)
    args = parser.parse_args()

    dataset = args.dataset_dir
    out_dir = args.output
    live_folders = set(args.live_folders)

    os.makedirs(out_dir, exist_ok=True)
    table1_path = os.path.join(out_dir, 'table1_detailed.csv')
    table2_path = os.path.join(out_dir, 'table2_counts.csv')
    gt_path = os.path.join(out_dir, 'ground_truth.csv')

    counts = Counter()
    total = 0
    inferred_attack_types = set()

    for dirpath, dirnames, filenames in os.walk(dataset):
        for fn in filenames:
            if not is_media_file(fn): continue
            rel = os.path.relpath(os.path.join(dirpath, fn), dataset)
            parts = Path(rel).parts
            fname_lower = fn.lower()
            is_live, inferred_set = infer_attack_types_from_parts(parts, live_folders)
            keyword_matches = classify_attack(parts, fname_lower)
            for a in inferred_set: inferred_attack_types.add(a)
            if is_live: counts['Live'] += 1
            else:
                counts['Non-live'] += 1
                combined = set(keyword_matches) | set(inferred_set)
                if combined: 
                    for m in combined: counts[m] += 1
                else: counts['Unspecified_NonLive'] += 1
            total += 1

    canonical = list(ATTACK_KEYWORDS.keys())
    extra_inferred = sorted(x for x in inferred_attack_types if x not in canonical)
    attack_columns = canonical + extra_inferred

    with open(table1_path, 'w', newline='', encoding='utf-8') as t1f, \
         open(gt_path, 'w', newline='', encoding='utf-8') as gtf:

        t1writer = csv.writer(t1f)
        gtwriter = csv.writer(gtf)
        header = ['image_id'] + attack_columns + ['file_type','is_video']
        t1writer.writerow(header)
        gtwriter.writerow(['path','label'])

        for dirpath, dirnames, filenames in os.walk(dataset):
            for fn in filenames:
                if not is_media_file(fn): continue
                rel = os.path.relpath(os.path.join(dirpath, fn), dataset)
                parts = Path(rel).parts
                ftype = file_type(fn)
                is_video_flag = 1 if ftype=='video' else 0
                fname_lower = fn.lower()
                is_live, inferred_set = infer_attack_types_from_parts(parts, live_folders)
                keyword_matches = classify_attack(parts, fname_lower)
                combined = set(keyword_matches) | set(inferred_set)
                row = [rel] + ['Yes' if at in combined else 'No' for at in attack_columns] + [ftype,is_video_flag]
                t1writer.writerow(row)
                gtwriter.writerow([rel, 1 if is_live else 0])

    with open(table2_path, 'w', newline='', encoding='utf-8') as t2f:
        writer = csv.writer(t2f)
        writer.writerow(['Attack_Type','Number_Sample'])
        special = {'Live','Non-live','Unspecified_NonLive'}
        attack_keys = sorted(k for k in counts.keys() if k not in special)
        for at in attack_keys:
            n = counts.get(at,0)
            if n>=args.min_sample_report: writer.writerow([at,n])
        if counts.get('Unspecified_NonLive',0) >= args.min_sample_report:
            writer.writerow(['Unspecified_NonLive',counts.get('Unspecified_NonLive',0)])
        writer.writerow(['Live',counts.get('Live',0)])
        writer.writerow(['Non-live',counts.get('Non-live',0)])
        writer.writerow(['Total',total])

    print('Wrote:',table1_path)
    print('Wrote:',table2_path)
    print('Wrote:',gt_path)

if __name__=='__main__':
    main()
