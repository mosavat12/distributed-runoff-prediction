import numpy as np
import pandas as pd

filepath = '/icebox/data/shares/mh2/mosavat/Distributed/train_inputs'
basin = '0510020304'

data = np.load(f'{filepath}/{basin}.npy')

df = pd.DataFrame(data[:, :, 0, :].mean(axis=(0, 1)), columns= ['prcp'])

df.to_csv(f'/icebox/data/shares/mh2/mosavat/Distributed/{basin}.csv', index = False)