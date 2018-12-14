from scipy.io import arff
import pandas as pd
import os
import sys

d = sys.argv[1]
for dataset_name in os.listdir(d):
    file_name = "{}.arff".format(os.path.join(d, dataset_name, dataset_name))
    try:
        data, meta = arff.loadarff(file_name)
        df = pd.DataFrame(data)
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values.astype(int)
        classes = df.iloc[:, -1].nunique()
        if classes in range(3, 10) and \
           x.shape[0] in range(0, 1000) and \
           x.shape[1] in range(0, 600):
            print("\"{}\", # shape:{}, classes:{}".format(
                dataset_name, x.shape, classes))
    except Exception:
        pass
