from rltf2.utils.visualization import draw_opt_percent_metric_hist
import json


def main():
    dir_path = '/results/new_method/DIAYN/DIAYN_Hopper-v2-cpprb-dup_resu/logs'
    tracker_history_path = '/results/new_method/DIAYN/DIAYN_Hopper-v2-cpprb-dup_resu/logs/train_tracker.json'

    colors = [(31/256, 119/256, 180/256), (255/256, 127/256, 14/256), (44/256, 160/256, 44/256), (214/256, 39/256, 40/256)]
    # Must start with minimal step / metric!
    bins_metric = [0, 500, 1001, 2001, 3001]
    bins_steps = [i * 50000 for i in range(0, 45)]
    if len(bins_steps) <= 25:
        x_dist = 2
    elif len(bins_steps) <= 50:
        x_dist = 4
    else:
        x_dist = 8
    metric = 'proj_returns'
    if len(colors) != len(bins_metric) - 1:
        colors = None

    tracker_dict = json.load(open(tracker_history_path, 'r'))
    opt_history = tracker_dict['options_history']

    draw_opt_percent_metric_hist(
        bins_metric=bins_metric,
        bins_steps=bins_steps,
        metric=metric,
        option_history_dict=opt_history,
        colors=colors,
        store_dir=dir_path,
        x_dist=x_dist
    )


if __name__ == '__main__':
    main()
