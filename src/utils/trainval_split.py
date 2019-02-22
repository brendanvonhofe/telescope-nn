import shutil
import os
import random

def main():
    fg_fns = os.listdir('data/processed/train/fg')
    bg_fns = os.listdir('data/processed/train/bg')
    random.shuffle(fg_fns)
    random.shuffle(bg_fns)

    for i in range(48):
        fn = fg_fns[i]
        shutil.move('data/processed/train/fg/'+fn, 'data/processed/val/fg/'+fn)
        shutil.move('data/processed/train/mattes/'+fn, 'data/processed/val/mattes/'+fn)

    for i in range(8261):
        fn = bg_fns[i]
        shutil.move('data/processed/train/bg/'+fn, 'data/processed/val/bg/'+fn)

if(__name__ == '__main__'):
    main()