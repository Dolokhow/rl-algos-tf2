import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


def binned_training_dynamics_plt(binned_metric_performance, bins_metric, bins_steps, store_dir, chart_name=None, colors=None, x_dist=2):
    bin_colors = []
    if colors is None or len(bins_metric) - 1 != len(colors):
        for index in range(len(bins_steps)):
            bin_colors.append(get_cmap(n=index, name='hsv'))
    else:
        bin_colors = colors
    bar_range = range(len(bins_steps) - 1)
    binned_metric_performance = np.array(binned_metric_performance)

    bottoms = np.zeros(len(bar_range))
    for met_ind in range(len(bins_metric) - 1):
        met_end_ind = met_ind + 1
        color = bin_colors[met_ind]
        name = str(bins_metric[met_ind]) + ' <= rew' + ' < ' + str(bins_metric[met_end_ind])
        plt.bar(bar_range, binned_metric_performance[:, met_ind], bottom=bottoms, color=color,
                edgecolor='white', width=1.0, label=name)
        bottoms += (binned_metric_performance[:, met_ind]).transpose()
    labels = []
    for step in bins_steps[1:]:
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
    if chart_name is None:
        chart_name = 'graph'
    plt.savefig(os.path.join(store_dir, 'opt_percent_' + chart_name + '_stacked_bar_plot.png'),
                dpi=600, bbox_inches='tight')
