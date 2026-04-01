import numpy as np, os
vf_dir = '/home/simuser/ws/data/value_functions'
for f in sorted(os.listdir(vf_dir)):
    if not f.endswith('.npz'): continue
    data = np.load(os.path.join(vf_dir, f), allow_pickle=True)
    keys = list(data.keys())
    shape = data['values'].shape if 'values' in keys else 'N/A'
    params = {}
    if 'params' in keys:
        try: params = data['params'].item()
        except: params = str(data['params'])
    print(f'{f}: shape={shape}, params={params}')
