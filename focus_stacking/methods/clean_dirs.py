import glob, os, shutil

dirs = glob.glob('/n/fs/3d-indoor/macro_data/focus_stack/bike1/*')
print(dirs)

for d in dirs:
    for focus in ['wide', 'narrow']:
        os.makedirs(f'{d}/{focus}/images', exist_ok=True)
        ims = glob.glob(f'{d}/{focus}/*.JPG') 
        im_names = [os.path.basename(im) for im in ims]
        for im_name in im_names:
           shutil.move(f'{d}/{focus}/{im_name}', f'{d}/{focus}/images/{im_name}')
