import numpy as np

from abc import ABC, abstractmethod

from collections import defaultdict

from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

from sklearn.utils.validation import check_is_fitted
from wildboar.tree import ShapeletTreeClassifier
from wildboar.distance import matches, distance


def shape_transform_(s, x, i, theta):
    """
    :param s: the shapelet
    :param x: the time series
    :param i: the matching location
    :param theta: the desired distance
    :returns: the shapelet with `theta`-distance
    """
    x_match = x[i:(i + s.shape[0])]
    v = (x_match - s)  # + np.finfo(np.float64).eps
    norm_v = np.linalg.norm(v)

    if norm_v == 0:
        x_match = np.random.uniform(s.shape)
        v = (x_match - s)
        norm_v = np.linalg.norm(v)

    return s + v / norm_v * theta


def extract_paths_(node, d, classes_, path=None):
    """Extract all paths categorized by class

    :param node: the root node of a tree
    :param d: the dictionary that stores the paths
    :param classes_: the classes
    :param path: list of paths (default: None)
    :returns: None
    """
    if path is None:
        path = []

    if node.is_leaf:
        d[classes_[np.argmax(node.proba)]].append(path)
    else:
        left_path = path.copy()
        left_path.append(("<=", node.shapelet.array, node.threshold))
        extract_paths_(node.left, d, classes_, left_path)

        right_path = path.copy()
        right_path.append((">", node.shapelet.array, node.threshold))
        extract_paths_(node.right, d, classes_, right_path)


class LabelTransformer(ABC):
    @abstractmethod
    def fit(self, x, y, to_label):
        pass

    @abstractmethod
    def transform(self, x):
        pass

    @abstractmethod
    def score(self, x, y):
        pass


class NearestNeighbourLabelTransformer(LabelTransformer):
    def __init__(self,
                 n_neighbors=1,
                 n_clusters="auto",
                 metric="euclidean",
                 random_state=10):
        self.nn_ = None
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, x, y, to_label):
        if self.n_clusters == "auto":
            self.n_clusters = x.shape[0] // self.n_neighbors

        self.classes_, y = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        to_idx = np.argwhere(self.classes_ == to_label)[0][0]
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state)

        self.kmeans_.fit(x)
        center_majority = np.zeros([self.kmeans_.n_clusters, n_classes])
        for l, c in zip(self.kmeans_.labels_, y):
            center_majority[l, c] += 1

        center_prob = center_majority / np.sum(
            center_majority, axis=1).reshape(-1, 1)

        majority_class = center_prob[:, to_idx] > (1.0 / n_classes)
        maximum_class = center_majority[:, to_idx] >= (
            self.n_neighbors // n_classes) + 1

        cluster_centers = self.kmeans_.cluster_centers_
        majority_centers = cluster_centers[majority_class & maximum_class, :]
        self.cluster_centers_ = majority_centers
        self.nn_ = NearestNeighbors(1, metric=self.metric)
        self.nnc_ = KNeighborsClassifier(self.n_neighbors, metric=self.metric)
        if self.cluster_centers_.shape[0] > 0:
            self.nn_.fit(self.cluster_centers_)
        self.nnc_.fit(x, y)
        self.to_label_ = to_label
        self.predictions_ = 1
        self.pruned_ = 0
        return self, self.kmeans_

    def transform(self, x):
        check_is_fitted(self, ["nn_", "cluster_centers_"])
        if self.cluster_centers_.shape[0] > 0:
            closest = self.nn_.kneighbors(x, return_distance=False)
            return self.cluster_centers_[closest[:, 0]]
        else:
            return np.full(x.shape, np.nan)

    def score(self, x, y):
        return np.sum(self.predict(x) == y) / x.shape[0]

    def predict(self, x):
        return self.classes_[self.nnc_.predict(x)]


class GreedyTreeLabelTransform(LabelTransformer):
    def __init__(self,
                 n_shapelets=100,
                 min_shapelet_size=0,
                 max_shapelet_size=1,
                 n_estimators=100,
                 n_jobs=8,
                 epsilon=1,
                 batch_size=1,
                 random_state=10):
        self.paths_ = None
        self.epsilon = epsilon
        self.n_shapelets = n_shapelets
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.batch_size = batch_size

    def fit(self, x, y, to_label):
        tree = ShapeletTreeClassifier(
            n_shapelets=self.n_shapelets,
            metric="euclidean",
            min_shapelet_size=self.min_shapelet_size,
            max_shapelet_size=self.max_shapelet_size,
        )
        self.ensemble_ = BaggingClassifier(
            base_estimator=tree,
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            n_jobs=self.n_jobs,
        )
        self.to_label_ = to_label
        self.ensemble_.fit(x, y)
        self.paths_ = defaultdict(list)
        for base_estimator in self.ensemble_.estimators_:
            extract_paths_(base_estimator.root_node_, self.paths_,
                           self.ensemble_.classes_, [])
        self.paths_ = dict(self.paths_)

        # heuristics
        if self.batch_size < 1:
            self.ensemble_.n_jobs = 1
        return self

    def transform(self, x):
        check_is_fitted(self, "paths_")
        x_prime = np.empty(x.shape)
        predictions = 0
        pruned = 0
        for i in range(x.shape[0]):
            x_prime[i, :], pred, prune = self._transform_single(x[i, :])
            predictions += pred
            pruned += prune

        self.pruned_ = pruned / float(x.shape[0])
        self.predictions_ = predictions / float(x.shape[0])
        return x_prime

    def _transform_single(self, x_i):
        path_list = self.paths_[self.to_label_]
        x_prime = np.empty([len(path_list), x_i.shape[0]])
        for i, path in enumerate(path_list):
            x_i_prime = self._transform_single_path(x_i.copy(), path)
            x_prime[i, :] = x_i_prime

        cost = np.linalg.norm(x_prime - x_i, axis=1)
        cost_sort = np.argsort(cost)

        # If the cost of prediction didn't carry the overhead of
        # copying data to different cores due to the python
        # implementation of `ShapeletTreeClassifier` a `batch_size` of
        # 1 would be optimal. However, empirically half of the
        # conversions seems to be the fastest in practice for this
        # implementation.
        #
        # Note that the cost is ordered in increasing order; hence, if
        # a conversion is successful there can exist no other
        # successful transformation with lower cost.
        predictions = 0
        batch_size = round(x_prime.shape[0] * self.batch_size) + 1
        for i in range(0, len(cost_sort), batch_size):
            end = min(len(cost_sort), i + batch_size)
            cost_sort_i = cost_sort[i:end]
            predictions += cost_sort_i.shape[0]
            x_prime_i = x_prime[cost_sort_i, :]
            y_prime_i = self.ensemble_.predict(x_prime_i)
            condition_i = y_prime_i == self.to_label_
            x_prime_i = x_prime_i[condition_i]
            cost_i = cost[cost_sort_i[condition_i]]
            if x_prime_i.shape[0] > 0:
                min_cost_i = np.argmin(cost_i)
                min_x = x_prime_i[min_cost_i, :]
                return min_x, predictions / float(len(path_list)), 0

        return np.nan, 1, 0  # np.full(x_i.shape, np.nan)

    def _transform_single_path(self, x, path):
        for direction, shapelet, threshold in path:
            if direction == "<=":
                dist, location = distance(
                    shapelet, x, metric="euclidean", return_index=True)
                if dist > threshold:
                    impute_shape = shape_transform_(shapelet, x, location,
                                                    threshold - self.epsilon)
                    x[location:(location + len(shapelet))] = impute_shape
            else:
                locations = matches(shapelet, x, threshold, metric="euclidean")
                if locations is not None:
                    for location in locations:
                        impute_shape = shape_transform_(
                            shapelet, x, location, threshold + self.epsilon)
                        x[location:(location + len(shapelet))] = impute_shape
        return x

    def score(self, x, y):
        check_is_fitted(self, "ensemble_")
        return self.ensemble_.score(x, y)

    def predict(self, x):
        return self.ensemble_.predict(x)


class IncrementalTreeLabelTransform(GreedyTreeLabelTransform):
    def _transform_single_path(self, x, path):
        for direction, shapelet, threshold in path:
            if direction == "<=":
                dist, location = distance(
                    shapelet, x, metric="euclidean", return_index=True)
                if dist > threshold:
                    impute_shape = shape_transform_(shapelet, x, location,
                                                    threshold - self.epsilon)
                    x[location:(location + len(shapelet))] = impute_shape
            else:
                dist, location = distance(
                    shapelet, x, metric="euclidean", return_index=True)
                while dist - threshold < 0.0001:
                    impute_shape = shape_transform_(shapelet, x, location,
                                                    threshold + self.epsilon)
                    x[location:(location + len(shapelet))] = impute_shape
                    dist, location = distance(
                        shapelet, x, metric="euclidean", return_index=True)
        return x


def locked_iter(locked, start, end):
    """iterates over unlocked regions. `locked must be sorterd`

    :param locked: a list of tuples `[(start_lock, end_lock)]`
    :param start: first index
    :param end: last index
    :yield: unlocked regions
    """
    if len(locked) == 0:
        yield start, end
    for i in range(0, len(locked)):
        s, e = locked[i]
        if i == 0:
            start = 0
        if s - start > 0:
            yield start, (start + (s - start))
        start = e

        if i == len(locked) - 1 and end - start > 0:
            yield start, end


def in_range(i, start, end):
    return i > start and i <= end


def is_locked(start, end, locked):
    """Return true if location is locked

    :param pos: the position
    :param locked: list of locked positions
    :returns: true if locked
    """
    for s, e in locked:
        if in_range(s, start, end) or in_range(e, start, end):
            return True
    return False


class LockingIncrementalTreeLabelTransform(IncrementalTreeLabelTransform):
    def _transform_single(self, x_orig):
        path_list = self.paths_[self.to_label_]
        best_cost = np.inf
        best_x_prime = np.nan
        batch_size = round(len(path_list) * self.batch_size) + 1
        x_prime = np.empty((batch_size, x_orig.shape[0]))
        n_paths = len(path_list)
        i = 0
        prune = 0
        predictions = 0
        while i < n_paths:
            j = 0
            while j < batch_size and i + j < n_paths:
                path = path_list[i + j]
                x_prime_j = self._transform_single_path(
                    x_orig, path, best_cost)
                if x_prime_j is not None:
                    x_prime[j, :] = x_prime_j
                    j += 1
                else:
                    prune += 1
                    i += 1
            i += j
            if j > 0:
                predictions += j
                x_prime_pred = x_prime[:j, :]
                cond = self.ensemble_.predict(x_prime_pred) == self.to_label_
                x_prime_i = x_prime_pred[cond]
                if x_prime_i.shape[0] > 0:
                    cost = np.linalg.norm(x_prime_i - x_orig, axis=1)
                    argmin_cost = np.argmin(cost)
                    min_cost = cost[argmin_cost]
                    if min_cost < best_cost:
                        best_cost = min_cost
                        best_x_prime = x_prime_i[argmin_cost]

        return (best_x_prime, predictions / float(len(path_list)),
                prune / float(len(path_list)))

    def _transform_single_path(self, x_orig, path, best_cost=np.inf):
        x_prime = x_orig.copy()
        locked = []
        for direction, shapelet, threshold in path:
            if direction == "<=":
                dist, location = distance(
                    shapelet, x_prime, metric="euclidean", return_index=True)

                if dist > threshold and not is_locked(
                        location, location + len(shapelet), locked):
                    impute_shape = shape_transform_(
                        shapelet, x_prime, location, threshold - self.epsilon)

                    x_prime[location:(location + len(shapelet))] = impute_shape
                    locked.append((location, location + len(shapelet)))

                    cost = np.linalg.norm(x_prime - x_orig)
                    if cost >= best_cost:
                        return None

            else:
                dist, location = distance(
                    shapelet, x_prime, metric="euclidean", return_index=True)
                while (dist - threshold < 0.0001 and not is_locked(
                        location, location + len(shapelet), locked)):
                    impute_shape = shape_transform_(
                        shapelet, x_prime, location, threshold + self.epsilon)

                    x_prime[location:(location + len(shapelet))] = impute_shape
                    locked.append((location, location + len(shapelet)))

                    cost = np.linalg.norm(x_prime - x_orig)
                    if cost >= best_cost:
                        return None

                    dist, location = distance(
                        shapelet,
                        x_prime,
                        metric="euclidean",
                        return_index=True)
        return x_prime
