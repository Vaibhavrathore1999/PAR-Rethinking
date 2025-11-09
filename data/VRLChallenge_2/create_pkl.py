"""
create_pkl.py

Converts your custom multi-attribute dataset into a PA100K-style EasyDict pickle
(e.g. dataset_all.pkl) and writes helper files. Designed to be dropped into your
repo's `data/` directory and run.

Assumptions about your dataset (based on your message):
- images/ contains 701 image files (relative to this script).
- labels.txt contains attribute names (one per line OR a single header line - script supports both forms).
- train.txt contains rows: "image_name <49-length binary vector>"
  Example row: `000001.jpg 1 0 0 1 0 ... 0` (space or comma separated).

Features of this script:
- Loads your labels and train.txt (or optionally val.txt/test.txt if provided).
- If only one split file is provided, it will create val/test splits automatically (default 70/15/15).
- Produces dataset_all.pkl matching PA100K's structure with keys:
  description, reorder, root, image_name, label, attr_name, label_idx, partition, weight_train, weight_trainval
- Uses easydict.EasyDict if available; otherwise falls back to a minimal EasyDict class.

Usage:
    python convert_pa100k_style_dataset.py --images images/ --labels labels.txt --input-split train.txt \
        --output dataset_all.pkl --root /absolute/path/to/images

If you already have val.txt and/or test.txt, pass them via --val-split and --test-split.

"""

import os
import argparse
import pickle
import numpy as np
from collections import defaultdict

# Try to use easydict if installed; otherwise define a small fallback that behaves similarly
try:
    from easydict import EasyDict
except Exception:
    class EasyDict(dict):
        """A tiny EasyDict fallback allowing attribute access (d.key) and dict access (d['key'])."""
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)
        def __setattr__(self, name, value):
            self[name] = value


def read_attr_names(path):
    """Read attribute names from labels.txt.
    Supports two formats:
      - One attribute name per line (49 lines)
      - Single header line with 49 names separated by whitespace or commas
    """
    with open(path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # If there's exactly 49 lines, assume one per line
    if len(lines) == 49:
        return lines

    # If first line contains many tokens, split it
    first = lines[0]
    if ',' in first:
        toks = [t.strip() for t in first.split(',') if t.strip()]
    else:
        toks = [t.strip() for t in first.split() if t.strip()]

    if len(toks) == 49:
        return toks

    # Fallback: return all tokens across all lines
    toks = []
    for l in lines:
        if ',' in l:
            toks.extend([t.strip() for t in l.split(',') if t.strip()])
        else:
            toks.extend([t.strip() for t in l.split() if t.strip()])
    if len(toks) == 49:
        return toks

    raise ValueError(f"Could not parse 49 attribute names from {path}. Found {len(toks)} tokens.")


def read_split_file(path, expected_attrs=None):
    """Read a split file like train.txt which is: image_name <49 values>
    Returns list of (image_name, label_vector)
    """
    items = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # allow comma or whitespace separation
            if ',' in line:
                parts = [p.strip() for p in line.split(',') if p.strip()]
            else:
                parts = [p.strip() for p in line.split() if p.strip()]
            if len(parts) < 2:
                raise ValueError(f"Bad line in {path} at {i+1}: '{line}'")
            img = parts[0]
            vals = parts[1:]
            # If values are in a single token like '10101', split characters
            if len(vals) == 1 and len(vals[0]) == 49 and all(c in '01' for c in vals[0]):
                vals = list(vals[0])
            if expected_attrs is not None and len(vals) != expected_attrs:
                raise ValueError(f"Line {i+1} in {path} has {len(vals)} attributes but expected {expected_attrs}")
            vec = np.array([int(x) for x in vals], dtype=np.int8)
            items.append((img, vec))
    return items


def compute_class_weights(labels_array):
    # labels_array shape: (N, C)
    N, C = labels_array.shape
    # frequency = positive fraction
    pos = labels_array.sum(axis=0).astype(np.float32)
    freq = pos / float(N)
    # weight like in PA100K: maybe 1 - freq? but we'll follow example values: weight = freq? We'll store 1 - freq
    weight = 1.0 - freq
    return weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', default='/janaki/backup/users/student/pg/pg23/vaibhav.rathore/datasets/VRLChallenge_2/images', help='Image directory')
    parser.add_argument('--labels', default='/janaki/backup/users/student/pg/pg23/vaibhav.rathore/datasets/VRLChallenge_2/labels.txt', help='Attribute names file')
    parser.add_argument('--input-split', default='/janaki/backup/users/student/pg/pg23/vaibhav.rathore/datasets/VRLChallenge_2/train.txt', help='Primary split file (required)')
    parser.add_argument('--val-split', default=None, help='Optional val split file')
    parser.add_argument('--test-split', default=None, help='Optional test split file')
    parser.add_argument('--output', default='dataset_all.pkl', help='Output pickle file')
    parser.add_argument('--root', default=None, help='Root absolute path saved inside pickle (optional)')
    parser.add_argument('--random-seed', type=int, default=1234)
    args = parser.parse_args()

    np.random.seed(args.random_seed)

    if not os.path.exists(args.labels):
        raise FileNotFoundError(f"labels file not found: {args.labels}")
    attr_names = read_attr_names(args.labels)
    C = len(attr_names)
    print(f"Found {C} attributes")

    # Read provided splits
    train_items = read_split_file(args.input_split, expected_attrs=C)
    val_items = read_split_file(args.val_split, expected_attrs=C) if args.val_split else []
    test_items = read_split_file(args.test_split, expected_attrs=C) if args.test_split else []

    # If val/test missing, split from train_items
    all_items = train_items + val_items + test_items
    # ensure unique image names
    names = [n for n, _ in all_items]
    if len(names) != len(set(names)):
        print("Warning: duplicate image names found across split files; duplicates will be kept as-is")

    if not val_items or not test_items:
        # create combined array of items and random split
        print("Creating val/test splits automatically (70/15/15) from provided items")
        items = train_items  # use primary list
        N = len(items)
        img_names = [x[0] for x in items]
        labels = np.stack([x[1] for x in items], axis=0)
        # For stratification approximation, use sum of labels per sample
        stratify = labels.sum(axis=1)
        from sklearn.model_selection import train_test_split
        train_idx, rest_idx = train_test_split(
            np.arange(N), test_size=0.30, random_state=args.random_seed
        )
        val_idx, test_idx = train_test_split(
            rest_idx, test_size=0.5, random_state=args.random_seed
        )


        train_list = [items[i] for i in train_idx]
        val_list = [items[i] for i in val_idx]
        test_list = [items[i] for i in test_idx]
    else:
        train_list = train_items
        val_list = val_items
        test_list = test_items

    # Build unified lists in order: train + val + test (this mirrors some dataset conventions)
    # image_name = [x[0] for x in (train_list + val_list + test_list)]
    image_name = []
    for x, _ in (train_list + val_list + test_list):
        name = x
        if not name.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = name + ".jpg"
        image_name.append(np.str_(name))   # ✅ this makes it np.str_ type
    label = np.stack([x[1] for x in (train_list + val_list + test_list)], axis=0).astype(np.int8)

    # Construct partition indices (0-based indices into image_name list)
    Ntrain = len(train_list)
    Nval = len(val_list)
    Ntest = len(test_list)

    partition = dict()
    partition['train'] = np.arange(0, Ntrain, dtype=np.int32)
    partition['val'] = np.arange(Ntrain, Ntrain + Nval, dtype=np.int32)
    partition['test'] = np.arange(Ntrain + Nval, Ntrain + Nval + Ntest, dtype=np.int32)
    # Use np.concatenate to combine numpy arrays
    partition['trainval'] = np.concatenate((partition['train'], partition['val']))
    # label_idx: for compatibility, provide 'eval' mapping (here we keep identity)
    # label_idx: provide 'all' (for training) and 'eval' (for eval)
    label_idx = {'all': list(range(C)), 'eval': list(range(C))}


    weight_train = compute_class_weights(label[partition['train'], :]) if len(partition['train']) > 0 else compute_class_weights(label)
    weight_trainval = compute_class_weights(label[np.array(partition['trainval']) if len(partition['trainval'])>0 else np.arange(label.shape[0]), :])

    # Build EasyDict structure
    data = EasyDict()
    data.description = 'VRLChallenge_2'
    data.reorder = 'group_order'
    data.root = args.root if args.root is not None else os.path.abspath(args.images)
    data.image_name = np.array(image_name, dtype=np.str_)
    data.label = label
    # data.attr_name = attr_names
    data.attr_name = np.array([np.str_(x) for x in attr_names], dtype=np.str_)
    data.label_idx = label_idx
    data.partition = partition
    data.weight_train = weight_train
    data.weight_trainval = weight_trainval

    # Save
    with open(args.output, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote pickle to {args.output}")
    print(f"Num images: {len(image_name)}, splits: train={len(partition['train'])}, val={len(partition['val'])}, test={len(partition['test'])}")


if __name__ == '__main__':
    main()
