import yaml
import os
import logging
import ntpath
import shutil
from time import time
import tensorflow as tf
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def get_logger(name, log_dir=None, level=logging.INFO):
    logger = logging.getLogger(name=name)
    logger.setLevel(level=level)
    formatter = logging.Formatter(fmt=' %(name)s :: %(levelname)-8s :: %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        filename = os.path.join(log_dir, name + str(time()) + '.log')
        file_handler = logging.FileHandler(filename, 'a')
        file_handler.setLevel(level=level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def yaml_to_dict(file_path):
    if not os.path.isfile(file_path):
        logger.warning('No file found at specified path.')
        return None
    with open(file=file_path, mode='r') as file:
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        yaml_dict = yaml.load(file, Loader=loader)
    return yaml_dict


def split_path(file_path):
    path_to_file, file_name = ntpath.split(file_path)
    base_name, ext = os.path.splitext(file_name)
    return path_to_file, file_name, base_name, ext


def check_create_dir(dir_path):
    if not os.path.isdir(dir_path):
        if os.path.exists(dir_path):
            os.mkdir(path=dir_path)
            return True
        else:
            logger.warning('Directory path not valid.')
            return False
    else:
        return True


def create_dir(path, unique_tag=True):
    if os.path.isdir(path):
        if unique_tag is True:
            path = tag_name(path=path)
        else:
            return False, None
    os.mkdir(path=path)
    return True, path


def tag_name(path):
    unique_tag = str(time())
    path_to_dst_file, _, base_file_name, file_ext = split_path(file_path=path)
    tag_name = base_file_name + '_' + unique_tag + file_ext
    return os.path.join(path_to_dst_file, tag_name)


def copy_file(src_path, dst_path, force=False, unique_tag=True):
    if not os.path.isfile(src_path):
        logger.error('No file found at specified source path.')
    if os.path.isfile(dst_path):
        if force is False:
            if unique_tag is False:
                logger.warning('Destination path file already exists. File NOT overwritten.')
                return False, None
            else:
                dst_path = tag_name(path=dst_path)
                shutil.copyfile(src=src_path, dst=dst_path)
                logger.warning('Destination path file already exists. File tagged with a timestamp.')
                return True, dst_path
        else:
            shutil.copyfile(src=src_path, dst=dst_path)
            logger.warning('Destination path file already exists. File overwritten.')
            return True, dst_path

    shutil.copyfile(src=src_path, dst=dst_path)
    return True, dst_path


def tensorboard_structured_summaries(writer, summaries, step):
    with writer.as_default():
        for summary in summaries:

            s_name = summary[0]
            if tf.is_tensor(s_name):
                s_name = s_name.numpy().decode('UTF-8')

            s_type = summary[1]
            if tf.is_tensor(s_type):
                s_type = s_type.numpy().decode('UTF-8')

            s_value = summary[2]
            if tf.is_tensor(s_value):
                s_value = s_value.numpy()

            if s_type == 'scalar':
                tf.summary.scalar(name=s_name, data=s_value, step=step)
        writer.flush()


def configure_tf_checkpoints(policy, dir_path):
    checkpoint = tf.train.Checkpoint(root=policy)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, directory=dir_path, max_to_keep=5)
    return checkpoint, checkpoint_manager


