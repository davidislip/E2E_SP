import numpy as np
import pickle
import os


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.default_rng(seed=42).permutation(len(a))
    return a[p], b[p]


def load_search_params(cache, experiment_folder, name):
    with open(cache + experiment_folder + name + "/search.pkl", 'rb') as f:
        search = pickle.load(f)
    with open(cache + experiment_folder + name + "/params.pkl", 'rb') as f:
        best_params = pickle.load(f)
    return search, best_params


from sklearn.model_selection import RandomizedSearchCV


def test_and_save_config(net, ps, X_train_scaled, y, params, cache, experiment_folder, name, n_iter=2):
    clf = RandomizedSearchCV(net, params, n_iter=n_iter,
                             refit=False, cv=ps, verbose=2)

    search = clf.fit(X_train_scaled, y)

    isExist = os.path.exists(cache + experiment_folder + "/")

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(cache + experiment_folder + name + "/")
        print("The new directory is created!")

    # saving
    with open(cache + experiment_folder + name + "/params.pkl", 'wb') as f:
        pickle.dump(search.best_params_, f)

    # saving
    with open(cache + experiment_folder + name + "/search.pkl", 'wb') as f:
        pickle.dump(search, f)
