import os
import numpy as np
import pandas as pd

for nobj in [2, 4]:
  print('Consolidating ' + nobj + ' objective SA')
  if nobj==2:
    dir_data = '../../data/policy_simulation/2obj/'
  else:
    dir_data = '../../data/policy_simulation/4obj/'
  
  files = os.listdir(dir_data)
  
  for f in files:
    try:
      mi_dict = pd.read_pickle(dir_data + f)
      polnum = int(f.split('.')[0])
      if 'mi_df' not in locals():
        mi_df = pd.DataFrame(mi_dict, index=[polnum])
      else:
        mi_df = mi_df.append(pd.DataFrame(mi_dict, index=[polnum]))
    except:
      print('Error with ' + f)
  
  mi_df.to_csv(dir_data + 'mi_combined.csv')





