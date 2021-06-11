from rltf2.utils.metric_inspection import binned_opt_training_dynamics
from rltf2.utils.visualization_utils import binned_training_dynamics_plt
import json


def main():
    dir_path = '/Users/djordjebozic/ML/personal/RL/rl_projects/algos/results/verified/Hopper/DIAYN_Hopper-v2-cpprb-paper-replica/logs'
    tracker_history_path = '/Users/djordjebozic/ML/personal/RL/rl_projects/algos/results/verified/Hopper/DIAYN_Hopper-v2-cpprb-paper-replica/logs/train_tracker.json'

    colors = [(31/256, 119/256, 180/256), (255/256, 127/256, 14/256), (44/256, 160/256, 44/256), (214/256, 39/256, 40/256)]
    # Must start with minimal step / metric!
    bins_metric = [0, 500, 1001, 1101, 2200]
    # Be super careful to cover only the actual steps agent performed.
    bins_steps = [i * 50000 for i in range(0, 41)]
    if len(bins_steps) <= 25:
        x_dist = 2
    elif len(bins_steps) <= 50:
        x_dist = 4
    else:
        x_dist = 8

    if len(colors) != len(bins_metric) - 1:
        colors = None

    tracker_dict = json.load(open(tracker_history_path, 'r'))
    opt_history = tracker_dict['options_history']
    metric_percentages = binned_opt_training_dynamics(
        option_history=opt_history,
        bins_metric=bins_metric,
        bins_steps=bins_steps,
        metric='proj_returns'
    )
    binned_training_dynamics_plt(
        binned_metric_performance=metric_percentages,
        bins_metric=sorted(bins_metric),
        bins_steps=sorted(bins_steps),
        colors=colors,
        store_dir=dir_path,
        chart_name='proj_returns_2',
        x_dist=x_dist
    )


if __name__ == '__main__':
    main()
