import numpy as np
import pickle
import os
import matplotlib.pyplot as plt


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


def search_boxplot_by_column(results_df, param_to_plot):
    plt.rcParams.update({'font.size': 6})

    param_values = results_df[param_to_plot]
    mean_test_scores = results_df.mean_test_score
    mean_fit_times = results_df.mean_fit_time
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 2))
    ax1.boxplot([mean_test_scores[param_values == value] for value in np.sort(param_values.unique())],
                labels=np.sort(param_values.unique()))
    ax1.set_title(f'Boxplot of Mean Test Scores by {param_to_plot}')
    ax1.set_xlabel(param_to_plot)
    ax1.set_ylabel('Test Score (MMD)')

    ax2.boxplot([mean_fit_times[param_values == value] for value in np.sort(param_values.unique())],
                labels=np.sort(param_values.unique()))
    ax2.set_title(f'Boxplot of Mean Fit Time by {param_to_plot}')
    ax2.set_xlabel(param_to_plot)
    ax2.set_ylabel('Mean Fit Time')


def print_best(search):
    print("best params ", search.best_params_)
    print("best score ", search.cv_results_['mean_test_score'][search.best_index_])
    print("train time ", search.cv_results_['mean_fit_time'][search.best_index_])


def plot_stacked_scenarios(ax, stacked_scenario, filtered_contexts, asset=0,
                           color='orange', alpha=0.05,
                           Limit=100, title_string='',
                           label='', linestyle='-', y_label=None, x_label=None):
    periods, num_scen, num_series = stacked_scenario.shape
    context_pds, _ = filtered_contexts.shape
    repeated_contexts = np.repeat(filtered_contexts[:, None, :], repeats=num_scen, axis=1)
    # print(repeated_contexts.shape)
    full_returns = np.concatenate([repeated_contexts, stacked_scenario], axis=0)
    # print(full_returns.shape)

    if num_scen >= Limit:
        test = (1 + full_returns[:, :500, :]).cumprod(axis=0)
    else:
        test = (1 + full_returns[:, :, :]).cumprod(axis=0)

    test = np.concatenate([np.ones_like(test[0:1]), test], axis=0)
    # print(test.shape)
    ax.plot(np.arange(context_pds, len(test)), test[context_pds:, :, asset], alpha=alpha, color=color, label=label,
            linestyle=linestyle);

    ax.plot(np.arange(context_pds + 1), test[:context_pds + 1, 0, asset], alpha=alpha, color='black', label=label,
            linestyle='dashdot');

    ax.set_title(title_string)

    if y_label is not None:
        ax.set_ylabel(y_label)

    if x_label is not None:
        ax.set_xlabel(x_label)
