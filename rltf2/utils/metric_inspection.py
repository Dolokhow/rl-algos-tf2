import numpy as np


def sorted_opt_performance(options_history, metric, max_total_step=None, last_n_runs=10):
    # We'll be modifying options history, it should not reflect outside function call
    options_history = options_history.copy()
    min_sample = 1e7
    for key, _ in options_history.items():
        total_steps_arr = np.array(options_history[key]['total_step'])
        if max_total_step is not None:
            max_step_index = np.searchsorted(total_steps_arr, max_total_step)
            options_history[key]['episodes'] = options_history[key]['episodes'][:max_step_index]
            options_history[key]['total_step'] = options_history[key]['total_step'][:max_step_index]
            options_history[key]['num_steps'] = options_history[key]['num_steps'][:max_step_index]
            options_history[key]['rel_steps'] = options_history[key]['rel_steps'][:max_step_index]
            options_history[key]['returns'] = options_history[key]['returns'][:max_step_index - 1]
            options_history[key]['proj_returns'] = options_history[key]['proj_returns'][:max_step_index]
        else:
            max_step_index = total_steps_arr.shape[0]

        if max_step_index < min_sample:
            min_sample = max_step_index

    options_ranking = []
    last_n_runs = min(min_sample, last_n_runs)
    for key, _ in options_history.items():
        performance = options_history[key][metric][-last_n_runs:]
        average_performance = np.mean(performance)
        options_ranking.append((key, average_performance))

    options_ranking = sorted(options_ranking, key=lambda entry: entry[1], reverse=True)
    return options_ranking


def binned_opt_training_dynamics(option_history, bins_metric, bins_steps, metric):
    sorted_met_bins = sorted(bins_metric)
    sorted_steps_bins = sorted(bins_steps)
    total_options = len(option_history)

    avg_performance_per_binned_step = []

    for step_ind in range(len(sorted_steps_bins) - 1):
        start = sorted_steps_bins[step_ind]
        end = sorted_steps_bins[step_ind + 1]

        metric_in_step_bin = []
        for opt_id, data in option_history.items():
            end_steps = np.array(data['total_step'])
            step_indices = np.where((start <= end_steps) & (end_steps < end))[0]
            look_back_index = 0
            while step_indices.shape[0] == 0:
                look_back_index += 1
                if step_ind - look_back_index < 0:
                    break
                revised_start = sorted_steps_bins[step_ind - look_back_index]
                revised_end = sorted_steps_bins[step_ind - look_back_index + 1]
                step_indices = np.where((revised_start <= end_steps) & (end_steps < revised_end))[0]

            if step_indices.shape[0] > 0:
                metric_data = np.array(data[metric]).take(step_indices)
                mean_metric = np.mean(metric_data)
            else:
                mean_metric = 0
            metric_in_step_bin.append(mean_metric)
        metric_in_step_bin = np.array(metric_in_step_bin)

        metric_in_step_bin_percent = []
        for met_ind in range(len(sorted_met_bins) - 1):
            start_met = sorted_met_bins[met_ind]
            end_met = sorted_met_bins[met_ind + 1]
            rew_indices_count = np.where((start_met <= metric_in_step_bin) & (metric_in_step_bin < end_met))[0].shape[0]
            rew_index_percent = rew_indices_count / total_options
            metric_in_step_bin_percent.append(rew_index_percent)
        avg_performance_per_binned_step.append(metric_in_step_bin_percent)

    return avg_performance_per_binned_step
