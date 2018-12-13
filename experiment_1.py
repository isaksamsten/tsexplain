import sys
import csv
import time

import numpy as np
import pandas as pd

from collections import defaultdict

from scipy.io import arff

from tstransform.evaluation import failures
from tstransform.evaluation import differences
from tstransform.evaluation import cost

from tstransform.transform import NearestNeighbourLabelTransformer
from tstransform.transform import GreedyTreeLabelTransform
from tstransform.transform import IncrementalTreeLabelTransform
from tstransform.transform import LockingIncrementalTreeLabelTransform


def group_labels(y):
    tmp = defaultdict(list)
    for i, label in enumerate(y):
        tmp[label].append(i)

    label_index = {label: np.array(arr) for label, arr in tmp.items()}
    return dict(label_index)


two_class_datasets = [
    # "Computers",
    # "MoteStrain",
    # "Ham",
    # #    "FordA",
    # "DistalPhalanxOutlineCorrect",
    # "MiddlePhalanxOutlineCorrect",
    # "PhalangesOutlinesCorrect",
    # "Earthquakes",
    # "Lightning2",
    "GunPoint",
    "ItalyPowerDemand",
    "TwoLeadECG",
    # "ProximalPhalanxOutlineCorrect",
    # "ECG200",
    # "Herring",
    # "ToeSegmentation2",
    # "HandOutlines",
    # "ToeSegmentation1",
    # "WormsTwoClass",
    # "ECGFiveDays",
    # "Wine",
    # "BirdChicken",
    # "SonyAIBORobotSurface2",
    # #    "FordB",
    # "Strawberry",
    # "SonyAIBORobotSurface1",
    # "Coffee",
    # "Wafer",
    # "BeetleFly",
    # "Yoga",
]

result_writer = csv.writer(sys.stdout)
result_writer.writerow([
    "dataset",
    "method",
    "epsilon",
    "to_label",
    "cost",
    "failures",
    "differences",
    "predictions",
    "pruned_transform",
    "score",
    "total_n_transform",
    "total_n_test",
    "total_n_not_to_label",
    "transform_time",
])

train_fraction = 0.8
random_seed = 10

for dataset_name in two_class_datasets:
    rnd = np.random.RandomState(random_seed)

    data, meta = arff.loadarff(
        "TSC Problems/{0}/{0}.arff".format(dataset_name))
    df = pd.DataFrame(data)
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(int)

    idx = np.arange(x.shape[0])
    rnd.shuffle(idx)

    train_size = round(x.shape[0] * train_fraction)

    x_train = x[idx[:train_size], :]
    y_train = y[idx[:train_size]]

    x_test = x[idx[train_size:], :]
    y_test = y[idx[train_size:]]

    label_index = group_labels(y_test)

    print(
        "{} of shape: {}, and {} with labels {}".format(
            dataset_name,
            x_train.shape,
            x_test.shape,
            label_index.keys(),
        ),
        file=sys.stderr)
    print(
        "Label sizes: {}".format(
            [(lbl, len(lbl_idx)) for lbl, lbl_idx in label_index.items()]),
        file=sys.stderr)

    nn_trans = NearestNeighbourLabelTransformer(n_neighbors=1)
    greedy_e_trans = IncrementalTreeLabelTransform(
        epsilon=0.01,
        random_state=random_seed,
        n_shapelets=100,
        n_jobs=8,
        batch_size=0.05,
    )
    incremental_e_trans = LockingIncrementalTreeLabelTransform(
        epsilon=0.01,
        random_state=random_seed,
        n_jobs=8,
    )

    for to_label in label_index.keys():
        # nn_trans.fit(x_train, y_train, to_label)

        if greedy_e_trans.paths_ is None:
            greedy_e_trans.fit(x_train, y_train, to_label)
            incremental_e_trans.__dict__ = greedy_e_trans.__dict__

            # nn_score = nn_trans.score(x_test, y_test)
            e_score = greedy_e_trans.score(x_test, y_test)
        else:
            greedy_e_trans.to_label_ = to_label
            incremental_e_trans.to_label_ = to_label

        methods = {
            #                "NN": (nn_trans, nn_score),
            "IE": (greedy_e_trans, e_score),
            "LIE": (incremental_e_trans, e_score)
        }
        for name, (trans, score) in methods.items():
            for epsilon in [0.01, 0.05, 0.1, 0.5, 1, 5]:
                x_test_not_to = x_test[trans.predict(x_test) != to_label]
                t = time.time()
                trans.epsilon = epsilon
                x_prime = trans.transform(x_test_not_to)
                t = time.time() - t
                c = cost(x_prime, x_test_not_to)
                d = differences(x_prime, x_test_not_to, axis=1)
                f = failures(x_prime) / float(x_test_not_to.shape[0])
                result_writer.writerow([
                    dataset_name,
                    name,
                    epsilon,
                    to_label,
                    c,
                    f,
                    d,
                    trans.predictions_,
                    trans.pruned_,
                    score,
                    x_test_not_to.shape[0],
                    x_test.shape[0],
                    y_test[y_test != to_label].shape[0],
                    t * 1000,  # ms
                ])
