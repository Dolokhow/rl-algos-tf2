import numpy as np
import matplotlib.pyplot as plt
import os


def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


def draw_opt_percent_metric_hist(bins_metric, bins_steps, metric, option_history_dict, store_dir, colors=None, x_dist=2):
    sorted_met_bins = sorted(bins_metric)
    sorted_steps_bins = sorted(bins_steps)
    total_options = len(option_history_dict)

    metric_percentages = []

    for step_ind in range(len(sorted_steps_bins) - 1):
        start = sorted_steps_bins[step_ind]
        end = sorted_steps_bins[step_ind+1]

        metric_in_step_bin = []
        for opt_id, data in option_history_dict.items():
            end_steps = np.array(data['total_step'])
            step_indices = np.where((start <= end_steps) & (end_steps < end))[0]
            look_back_index = 1
            while step_indices.shape[0] == 0:
                if step_ind - look_back_index < 0:
                    break
                look_back_index += 1
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
            end_met = sorted_met_bins[met_ind+1]
            rew_indices_count = np.where((start_met <= metric_in_step_bin) & (metric_in_step_bin < end_met))[0].shape[0]
            rew_index_percent = rew_indices_count / total_options
            metric_in_step_bin_percent.append(rew_index_percent)
        metric_percentages.append(metric_in_step_bin_percent)

    bin_colors = []
    if colors is None or len(sorted_met_bins) - 1 != len(colors):
        for index in range(len(sorted_met_bins)):
            bin_colors.append(get_cmap(n=index, name='hsv'))
    else:
        bin_colors = colors
    bar_range = range(len(sorted_steps_bins) - 1)
    metric_percentages = np.array(metric_percentages)

    bottoms = np.zeros(len(bar_range))
    for met_ind in range(len(sorted_met_bins) - 1):
        met_end_ind = met_ind + 1
        color = bin_colors[met_ind]
        name = str(sorted_met_bins[met_ind]) + ' <= rew' + ' < ' + str(sorted_met_bins[met_end_ind])
        plt.bar(bar_range, metric_percentages[:, met_ind], bottom=bottoms, color=color,
                edgecolor='white', width=1.0, label=name)
        bottoms += (metric_percentages[:, met_ind]).transpose()
    labels = []
    for step in sorted_steps_bins[1:]:
        if step >= 1e6:
            if step % 1e6 == 0:
                step_str = str(int(step / 1e6)) + 'M'
            else:
                step_str = str(np.round(step / 1e6, 1)) + 'M'
        elif step >= 1e3:
            if step % 1e3 == 0:
                step_str = str(int(step / 1e3)) + 'K'
            else:
                step_str = str(np.round(step / 1e3, 1)) + 'K'
        else:
            step_str = step
        labels.append(step_str)
    tick_range = []
    label_range = []
    for index in range(len(bar_range)):
        if index % x_dist == 1:
            tick_range.append(bar_range[index])
            label_range.append(labels[index])
    plt.xticks(tick_range, label_range)
    plt.xlabel('steps')
    # Add a legend
    plt.legend(loc='best', ncol=1)
    # Show graphic
    plt.savefig(os.path.join(store_dir, 'opt_percent_' + metric + '_stacked_bar_plot.png'),
                dpi=600, bbox_inches='tight')


