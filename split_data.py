import os, random, shutil

def split_folder(src_folder, dst_root, splits=(0.7,0.2,0.1), seed=42):
    """
    src_folder: e.g. "spectrograms/ai"
    dst_root: e.g. "spectrograms"
    splits: train/val/test fractions summing to 1.0
    """
    random.seed(seed)
    fnames = [f for f in os.listdir(src_folder) if f.endswith('.png')]
    random.shuffle(fnames)
    
    n = len(fnames)
    n_train = int(splits[0]*n)
    n_val   = int(splits[1]*n)
    # remainder to test
    train_files = fnames[:n_train]
    val_files   = fnames[n_train:n_train+n_val]
    test_files  = fnames[n_train+n_val:]
    
    # helper to copy
    def copy_list(file_list, subset):
        dst_folder = os.path.join(dst_root, subset, os.path.basename(src_folder))
        os.makedirs(dst_folder, exist_ok=True)
        for fname in file_list:
            shutil.copy(os.path.join(src_folder, fname),
                        os.path.join(dst_folder, fname))
    
    copy_list(train_files, 'train')
    copy_list(val_files,   'val')
    copy_list(test_files,  'test')

# run for both classes
for label in ['ai','real']:
    split_folder(f'spectrograms/{label}', 'spectrograms', splits=(0.7,0.2,0.1))
