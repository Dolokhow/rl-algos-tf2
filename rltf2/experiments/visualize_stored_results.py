from rltf2.utils.visualization import draw_opt_percent_metric_hist
import json


def main():
    dir_path = '/Users/djordjebozic/ML/personal/RL/rl_projects/algos/results/new_method/DIAYN/DIAYN_InvertedPendulum-v2-cpprb/logs/'
    tracker_history_path = '/Users/djordjebozic/ML/personal/RL/rl_projects/algos/results/new_method/DIAYN/DIAYN_InvertedPendulum-v2-cpprb/logs/train_tracker.json'

    colors = [(31/256, 119/256, 180/256), (255/256, 127/256, 14/256)]
    # Must start with minimal step / metric!
    bins_metric = [0, 500, 1001]
    bins_steps = [i * 50000 for i in range(0, 21)]
    metric = 'proj_returns'

    tracker_dict = json.load(open(tracker_history_path, 'r'))
    opt_history = tracker_dict['options_history']

    draw_opt_percent_metric_hist(
        bins_metric=bins_metric,
        bins_steps=bins_steps,
        metric=metric,
        option_history_dict=opt_history,
        colors=colors,
        store_dir=dir_path
    )


if __name__ == '__main__':
    main()
