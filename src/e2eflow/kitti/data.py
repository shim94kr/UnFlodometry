import os
import sys

import numpy as np
import matplotlib.image as mpimg
import tensorflow as tf

from . import raw_records
from ..core.data import Data
from ..util import tryremove
from ..core.input import frame_name_to_num


def exclude_test_and_train_images(kitti_dir, exclude_lists_dir, exclude_target_dir,
                                  remove=False):
    to_move = []

    def exclude_from_seq(day_name, seq_str, image, view, distance=10):
        # image is the first frame of each frame pair to exclude
        seq_dir_rel = os.path.join(day_name, seq_str, view, 'data')
        seq_dir_abs = os.path.join(kitti_dir, seq_dir_rel)
        target_dir_abs = os.path.join(exclude_target_dir, seq_dir_rel)
        if not os.path.isdir(seq_dir_abs):
            print("Not found: {}".format(seq_dir_abs))
            return
        try:
            os.makedirs(target_dir_abs)
        except:
            pass
        seq_files = sorted(os.listdir(seq_dir_abs))
        image_num = frame_name_to_num(image)
        try:
            image_index = seq_files.index(image)
        except ValueError:
            return
        # assume that some in-between files may be missing
        start = max(0, image_index - distance)
        stop = min(len(seq_files), image_index + distance + 2)
        start_num = image_num - distance
        stop_num = image_num + distance + 2
        for i in range(start, stop):
            filename = seq_files[i]
            num = frame_name_to_num(filename)
            if num < start_num or num >= stop_num:
                continue
            to_move.append((os.path.join(seq_dir_abs, filename),
                            os.path.join(target_dir_abs, filename)))

    for filename in os.listdir(exclude_lists_dir):
        exclude_list_path = os.path.join(exclude_lists_dir, filename)
        with open(exclude_list_path) as f:
            for line in f:
                line = line.rstrip('\n')
                if line.split(' ')[0].endswith('_10'):
                    splits = line.split(' ')[-1].split('\\')
                    image = splits[-1]
                    seq_str = splits[0]
                    day_name, seq_name = seq_str.split('_drive_')
                    seq_name = seq_name.split('_')[0] + '_sync'
                    seq_str = day_name + '_drive_' + seq_name
                    exclude_from_seq(day_name, seq_str, image, 'image_02')
                    exclude_from_seq(day_name, seq_str, image, 'image_03')
    if remove:
        print("Collected {} files. Deleting...".format(len(to_move)))
    else:
        print("Collected {} files. Moving...".format(len(to_move)))

    for i, data in enumerate(to_move):
        try:
            src, dst = data
            print("{} / {}: {}".format(i, len(to_move) - 1, src))
            if remove:
                os.remove(src)
            else:
                os.rename(src, dst)
        except: # Some ranges may overlap
            pass

    return len(to_move)


class KITTIData(Data):
    KITTI_RAW_URL = 'http://kitti.is.tue.mpg.de/kitti/raw_data/'
    KITTI_2012_URL = 'http://kitti.is.tue.mpg.de/kitti/data_stereo_flow.zip'
    KITTI_2015_URL = 'http://kitti.is.tue.mpg.de/kitti/data_scene_flow.zip'

    dirs = ['data_stereo_flow', 'data_scene_flow', 'kitti_raw']

    def __init__(self, data_dir, stat_log_dir=None,
                 development=True, fast_dir=None):
        super().__init__(data_dir, stat_log_dir,
                         development=development,
                         fast_dir=fast_dir)

    def _fetch_if_missing(self):
        self._maybe_get_kitti_raw()
        self._maybe_get_kitti_2012()
        self._maybe_get_kitti_2015()

    def get_raw_dirs(self):
        top_dir = os.path.join(self.current_dir, 'kitti_raw')
        dirs = []
        dates = os.listdir(top_dir)
        for date in dates:
            date_path = os.path.join(top_dir, date)
            if os.path.isdir(date_path):
                extracts = os.listdir(date_path)
                for extract in extracts:
                    extract_path = os.path.join(date_path, extract)
                    if os.path.isdir(extract_path):
                        image_02_folder = os.path.join(extract_path, 'image_02/data')
                        image_03_folder = os.path.join(extract_path, 'image_03/data')
                        dirs.extend([image_02_folder, image_03_folder])
        return dirs

    def get_raw_intrinsics(self, filenames):
        # filenames : list('../data/kitti_raw/dates/dates_drives/image_cam/data/**.png', ..)
        intrinsics = []
        for file in filenames:
            top_dir = os.path.join(self.current_dir, 'kitti_raw')
            date = file.split('kitti_raw/')[1].split('/')[0]
            cam = file.split('image_')[1].split('/')[0]
            calib_file = os.path.join(top_dir, date, 'calib_cam_to_cam.txt')

            filedata = self.read_raw_calib_file(calib_file)
            P_rect = np.reshape(filedata['P_rect_' + cam], (3, 4))
            P_rect_tf = tf.constant(P_rect[:3,:3])
            intrinsics.append(P_rect_tf)
        return intrinsics

    def _maybe_get_kitti_2012(self):
        local_path = os.path.join(self.data_dir, 'data_stereo_flow')
        if not os.path.isdir(local_path):
            self._download_and_extract(self.KITTI_2012_URL, local_path)

    def _maybe_get_kitti_2015(self):
        local_path = os.path.join(self.data_dir, 'data_scene_flow')
        if not os.path.isdir(local_path):
            self._download_and_extract(self.KITTI_2015_URL, local_path)

    def _maybe_get_kitti_raw(self):
        base_url = self.KITTI_RAW_URL
        local_dir = os.path.join(self.data_dir, 'kitti_raw')
        records = raw_records.get_kitti_records(self.development)
        downloaded_records = False
          
        for i, record in enumerate(records):
            date_str = record.split("_drive_")[0]
            foldername = record + "_sync"
            date_folder = os.path.join(local_dir, date_str)
            if not os.path.isdir(date_folder):
                os.makedirs(date_folder)
            local_path = os.path.join(date_folder, foldername)
            if not os.path.isdir(local_path):
                url = base_url + record + "/" + foldername + '.zip'
                print(url)
                self._download_and_extract(url, local_dir)
                downloaded_records = True

            # Remove unused directories
            tryremove(os.path.join(local_path, 'velodyne_points'))
            tryremove(os.path.join(local_path, 'oxts'))
            tryremove(os.path.join(local_path, 'image_00'))
            tryremove(os.path.join(local_path, 'image_01'))

        if downloaded_records:
            print("Downloaded all KITTI raw files.")
            exclude_target_dir = os.path.join(self.data_dir, 'exclude_target_dir')
            exclude_lists_dir = '../files/kitti_excludes'
            excluded = exclude_test_and_train_images(local_dir, exclude_lists_dir, exclude_target_dir, remove=True)

    def read_raw_calib_file(self,filepath):
        # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """Read in a calibration file and parse into a dictionary."""
        data = {}

        with open(filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                        data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                        pass
        return data

    def scale_intrinsics(self, mat, sx, sy):
        out = np.copy(mat)
        out[0,0] *= sx
        out[0,2] *= sx
        out[1,1] *= sy
        out[1,2] *= sy
        return out
