import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from rltf2.utils.vector_ops import int_to_str

MAX_VID_WIDTH_PX = 1200
MAX_VID_HEIGHT_PX = 800


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
        step_str = int_to_str(n=step)
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


def load_video(path):
    cap = cv2.VideoCapture(path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(5))
    params = (width, height, fps)
    return cap, params


def create_video(path, code, params, extension=None):
    fourcc = cv2.VideoWriter_fourcc(*code)
    if extension is not None:
        if path.endswith(extension):
            pass
        else:
            path += extension
    out = cv2.VideoWriter(path, fourcc, params[2], (params[0], params[1]))
    return out


def paint_text(frame, text, color, position, boldness=2, size=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame = np.array(frame)
    cv2.putText(frame, text, position, font, size, color, boldness, cv2.LINE_AA)
    return frame


def merge_videos_to_matrix(video_paths, out_path, n_cols, auto_repeat=True):
    total_videos = len(video_paths)
    n_rows = np.int(np.ceil(total_videos / n_cols))

    # Assumes all videos have the same height and width
    cap, params = load_video(video_paths[0])
    cap.release()
    frame_w = params[0]
    frame_h = params[1]
    fps = params[2]
    aspect_ratio = int(frame_w / frame_h)

    # Resize dims to fit screen width
    potential_width = frame_w * n_cols
    if potential_width > MAX_VID_WIDTH_PX:
        reduce_factor = MAX_VID_WIDTH_PX / potential_width
        frame_w = np.int(np.floor(reduce_factor * frame_w))
        frame_h = np.int(frame_w / aspect_ratio)

    potential_height = frame_h * n_cols
    if potential_height > MAX_VID_HEIGHT_PX:
        reduce_factor = MAX_VID_HEIGHT_PX / potential_height
        frame_h = np.int(np.floor(reduce_factor * frame_h))
        frame_w = np.int(frame_h * aspect_ratio)

    caps = []
    frame_counts = []
    for video_path in video_paths:
        cap, params = load_video(path=video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        caps.append(cap)
        frame_counts.append(frame_count)
    for index in range(n_cols * n_rows - total_videos):
        # Add dummy caps for which we are using blank images
        caps.append(None)

    if os.path.isdir(out_path):
        name = str(n_rows) + '_by_' + str(n_cols) + '.mp4'
        out_path = os.path.join(out_path, name)
    elif not out_path.endswith('.mp4'):
        out_path = out_path + '.mp4'

    out = create_video(path=out_path, code='mp4v', params=[frame_w * n_cols, frame_h * n_rows, fps])
    max_frame_count = max(frame_counts)

    last_captured_frame_per_cap = [None for _ in caps]
    for index in range(max_frame_count):
        # Read one one frame from each video and pad if n_rows * n_cols != total_videos
        frames = []
        for c_index in range(len(caps)):
            cap = caps[c_index]
            if cap is not None:
                ret, frame = cap.read()
                if ret is True:
                    last_captured_frame_per_cap[c_index] = frame
                else:
                    if auto_repeat is False:
                        frame = last_captured_frame_per_cap[c_index]
                    else:
                        cap.release()
                        cap, _ = load_video(path=video_paths[c_index])
                        caps[c_index] = cap
                        ret, frame = cap.read()
            else:
                frame = np.uint8(np.zeros((frame_h, frame_w, 3)))
            frames.append(cv2.resize(frame, (frame_w, frame_h)))

        # Combine read frames into rows and columns
        row_vector = frames[0]
        row_vectors = []
        f_index = 1
        total_frames = len(frames)
        while f_index < total_frames:
            while f_index - len(row_vectors) * n_cols < n_cols:
                frame = frames[f_index]
                row_vector = np.concatenate((row_vector, frame), axis=1)
                f_index += 1
            row_vectors.append(row_vector)
            if f_index < total_frames:
                row_vector = frames[f_index]
            f_index += 1

        out_frame = row_vectors[0]
        for row_vector in row_vectors[1:]:
            out_frame = np.concatenate((out_frame, row_vector), axis=0)

        out.write(out_frame)

    for cap in caps:
        if cap is not None:
            cap.release()
    out.release()


if __name__ == '__main__':
    input_videos = [
        '/Users/djordjebozic/ML/personal/RL/rl_projects/algos/results/verified/Hopper/DIAYN_Hopper-v2-cpprb-paper-replica/vids/DIAYN_1.5M_opt_19_5.mp4',
        '/Users/djordjebozic/ML/personal/RL/rl_projects/algos/results/verified/Hopper/DIAYN_Hopper-v2-cpprb-paper-replica/vids/DIAYN_1.5M_opt_24_33.mp4',
        '/Users/djordjebozic/ML/personal/RL/rl_projects/algos/results/verified/Hopper/DIAYN_Hopper-v2-cpprb-paper-replica/vids/DIAYN_1.5M_opt_30_29.mp4',
        '/Users/djordjebozic/ML/personal/RL/rl_projects/algos/results/verified/Hopper/DIAYN_Hopper-v2-cpprb-paper-replica/vids/DIAYN_1.5M_opt_3_8.mp4',
        '/Users/djordjebozic/ML/personal/RL/rl_projects/algos/results/verified/Hopper/DIAYN_Hopper-v2-cpprb-paper-replica/vids/DIAYN_1.5M_opt_12_44.mp4',
        '/Users/djordjebozic/ML/personal/RL/rl_projects/algos/results/verified/Hopper/DIAYN_Hopper-v2-cpprb-paper-replica/vids/DIAYN_1.5M_opt_27_24.mp4',
        '/Users/djordjebozic/ML/personal/RL/rl_projects/algos/results/verified/Hopper/DIAYN_Hopper-v2-cpprb-paper-replica/vids/DIAYN_1.5M_opt_8_35.mp4'
    ]
    out_video = '/Users/djordjebozic/ML/personal/RL/rl_projects/algos/results/verified/Hopper/DIAYN_Hopper-v2-cpprb-paper-replica/vids/'
    merge_videos_to_matrix(
        video_paths=input_videos,
        out_path=out_video,
        n_cols=4,
        auto_repeat=True
    )
