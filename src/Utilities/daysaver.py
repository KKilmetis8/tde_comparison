import numpy as np
from src.Utilities.parser import parse

#%%
pre = '/home/s3745597/data1/TDE/'
args = parse()
fixes = np.arange(args.first, args.last + 1)
sim = args.name

days = []
for fix in fixes:
    day = np.loadtxt(f'{pre}{sim}/snap_{fix}/tbytfb_{fix}.txt')
    days.append(day)

savepath = f'{pre}/tde_comparison/data/days{sim}'
np.savetxt(savepath, days)
