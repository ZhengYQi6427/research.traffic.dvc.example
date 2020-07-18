import os
import cv2
import json
import time
import numpy as np
import pandas as pd
import multiprocessing

from copy import deepcopy
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment


class DetectionMeasurement:
    """
    This class can be used to measure object detection algorithm performance using metrics
     standardized by the VOC Challenge: http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf
     and refined by the COCO Consortium: http://cocodataset.org/#detection-eval

    Typical usage:
    - instantiate class
    - call get_all_metrics() to return mAP and AP50
    - ex:
        d = DetectionMeasurement(actual_data, predicted_data, 'maskrcnn')  # instantiate using default values for most params
        metrics = d.get_all_metrics()
        metrics['mAP']  # check out mean average precision
        metrics['mIoU']['car']  # check out mean IoU value for all matches in the 'car' class

    """

    def __init__(self, actuals, predictions, prediction_format, actuals_format='cvat', frame_numbers=None,
                 confidence_threshold=0.0, primary_iou_threshold=0.5, iou_thresholds=None,
                 recall_levels=np.linspace(0.0, 1.0, 11), test=False):
        """

        :param actuals: actual labels and bounding box measurements in
                        dict/json of computer vision annotation tool (cvat)
                        standard output format. See https://github.com/opencv/cvat
                        Note that object_ids must be numeric.
        :param predictions: predicted labels and bounding box measurements in the
                            standard output format of whatever model created them
                            For Supervisely outputs, this will be a path to a
                            directory with internal files named by frame number
        :param prediction_format: string identifying the format of the prediction data. Must be one of:
                                  ['maskrcnn', 'supervisely', 'zklab_measurement', 'tabular'].
                                   List of options can only be updated if _build_predictions() is updated to convert the data.
        :param actuals_format: string identifying the format of the actuals data. Must be one of:
                               ['cvat', 'zklab_measurement', 'tabular', 'supervisely'].
                               List of options can only be updated if _build_actuals() is updated to convert the data.
        :param frame_numbers: list of float, identifies the frames to look at in both the actuals and predictions
        :param confidence_threshold: float, minimum confidence level required to consider a detection proposal
        :param primary_iou_threshold: float value denoting lower bound for a valid Jaccard index (i.e.
                                      intersection over union) for an actual object and a predicted bounding box.
                                      Comparisons of predictions to actuals that yield deltas less than this
                                      value render the prediction an invalid identification of that actual.
                                      Default value is taken from VOC Challenge standard
        :param iou_thresholds: list of float values denoting all iou thresholds to use in computing mean average
                               precision (mAP), as done for COCO, where default is np.linspace(0.05, 0.95, 10)
                               when left empty, the list will consist only of the primary_iou_threshold
        :param recall_levels: list of float values denoting all recall levels to use in computing average precision
                              default comes from VOC; a second option is the COCO approach of np.linspace(0.0, 1.0, 101)
                              Note that when recall = 0.0, precision is always 1.0.
        :param test: bool, defaults to False. True only when instantiating strictly for testing purposes.
        """
        # initialize key variables
        self.results_df = pd.DataFrame()
        self.legal_matches = {}  # dict of float iou_threshold to tuple (float frame number, str class name) to (actual object_id, predicted object_id) to distance between actual and pred
        self.best_matches = {}  # dict of float iou_threshold to tuple (float frame number, str class name) to (actual object_id, predicted object_id) to distance between actual and pred
        self.misses = {}  # dict of float iou_threshold to tuple (float frame number, str class name) to list of object_ids
        self.false_positives = {}  # dict of float iou_threshold to tuple (float frame number, str class name) to list of object_ids

        self.class_names = ['bg', 'pedestrian', 'vehicle']
        self.total_objects_seen = 0  # count of object observances across all frames
        self._next_object_id = -1
        self.actuals = {}
        self.predictions = {}
        self.frame_numbers = []
        if frame_numbers:
            self.frame_numbers = list(set([float(x) for x in frame_numbers]))

        # numeric value to be used in the cost matrix of actual-prediction distances when their pairing is not valid
        # needs to be sufficiently high so as to eliminate the chance that this pairing is chosen in the cost minimization
        self.default_high = 100.0
        self.confidence_threshold = confidence_threshold

        if not test:
            # validate IoU & recall inputs
            self.iou_thresholds = [float(x) for x in iou_thresholds] if iou_thresholds else [
                float(primary_iou_threshold)]
            self.recall_levels = list(recall_levels)
            values = [primary_iou_threshold] + self.iou_thresholds + self.recall_levels
            for val in values:
                if not 0 <= val <= 1:
                    raise ValueError("All IoU thresholds and recall levels must be between 0 and 1.")
            self.recall_levels.sort()

            # params used to determine valid matches
            self.primary_iou_threshold = float(primary_iou_threshold)

            # extract relevant data about actuals
            self._build_actuals(actuals, actuals_format, frame_numbers)

            # extract relevant data about predictions
            self._build_predictions(predictions, prediction_format)

            # determine matches and errors
            start = time.time()
            self._get_matches_and_errors()
            end = time.time()
            print("Time to compute all matches and errors: {} seconds".format(end - start))

    def _get_next_object_id(self):
        """Provides a unique ID to be used for an object

        :return: float
        """
        self._next_object_id += 1.0
        return self._next_object_id

    def _build_actuals(self, actuals, actuals_format, frame_numbers):
        if actuals_format == 'cvat':
            # actuals = json.load(open(actuals))
            for obj in actuals['track']:
                class_name = obj['_label'].lower()
                if class_name not in self.class_names:
                    ##### hotfix for data from Summer 2019 and prior
                    if class_name in ['bus', 'car', 'truck', 'vehicle']:
                        class_name = 'vehicle'
                    else:
                        class_name = 'pedestrian'
                    ##### hotfix end
                for box in obj['box']:
                    self.total_objects_seen += 1
                    frame_number = float(box['_frame'])
                    if not frame_numbers or len(frame_numbers) == 0 or frame_number in self.frame_numbers:
                        box_data = {
                            'xtl': float(box['_xtl']),
                            'ytl': float(box['_ytl']),
                            'xbr': float(box['_xbr']),
                            'ybr': float(box['_ybr'])
                        }
                        if frame_number not in self.frame_numbers:
                            self.frame_numbers.append(frame_number)
                        if (frame_number, class_name) not in self.actuals:
                            self.actuals[(frame_number, class_name)] = {}
                        self.actuals[(frame_number, class_name)][self._get_next_object_id()] = deepcopy(box_data)

        elif actuals_format == 'zklab_measurement':
            for (frame_index, class_name), objects in actuals.items():
                if frame_index not in self.frame_numbers:
                    self.frame_numbers.append(frame_index)
                self.actuals[(frame_index, class_name)] = {}
                for _, object_data in objects.items():
                    self.actuals[(frame_index, class_name)][self._get_next_object_id()] = object_data
                    self.total_objects_seen += 1

        elif actuals_format == 'supervisely':
            # TODO the below assumes that filenames will always have the naming structure 'frame_XXXXX.png.json'
            # TODO (cont'd) update to use regex
            for file_name in os.listdir(actuals):
                frame_id = file_name[6:11]
                frame_number = float(frame_id)
                frame_preds = json.load(open(os.path.join(actuals, "frame_" + frame_id + ".png.json")))
                for object in frame_preds['objects']:
                    box_data = {
                        'xtl': float(object['points']['exterior'][0][0]),
                        'ytl': float(object['points']['exterior'][0][1]),
                        'xbr': float(object['points']['exterior'][1][0]),
                        'ybr': float(object['points']['exterior'][1][1])
                    }
                    class_name = object['classTitle'].lower()
                    if class_name != 'bg':
                        ##### hotfix for data from Summer 2019 and prior
                        if class_name in ['bus', 'car', 'truck', 'vehicle']:
                            class_name = 'vehicle'
                        else:
                            class_name = 'pedestrian'
                        ##### hotfix end
                        if frame_number not in self.frame_numbers:
                            self.frame_numbers.append(frame_number)
                        if (frame_number, class_name) not in self.actuals:
                            self.actuals[(frame_number, class_name)] = {}
                        self.actuals[(frame_number, class_name)][self._get_next_object_id()] = deepcopy(box_data)
                        self.total_objects_seen += 1

        elif actuals_format == 'tabular':
            for idx in actuals.index:
                frame_number = float(actuals.iloc[idx]['frame_number'])
                class_name = actuals.iloc[idx]['class_name']
                ##### hotfix for data from Summer 2019 and prior
                if class_name in ['bus', 'car', 'truck', 'vehicle']:
                    class_name = 'vehicle'
                else:
                    class_name = 'pedestrian'
                ##### hotfix end
                box_data = {
                    'xtl': float(actuals.iloc[idx]['xtl']),
                    'ytl': float(actuals.iloc[idx]['ytl']),
                    'xbr': float(actuals.iloc[idx]['xbr']),
                    'ybr': float(actuals.iloc[idx]['ybr'])
                }
                if (frame_number, class_name) not in self.actuals:
                    self.actuals[(frame_number, class_name)] = {}
                self.actuals[(frame_number, class_name)][self._get_next_object_id()] = deepcopy(box_data)
        else:
            raise NotImplementedError

    def _build_predictions(self, predictions, prediction_format):
        """Builds the dictionary of predictions in the following format:
        dict of float (frame number) to float (object_id) to
        string ('xtl', 'ytl', 'xbr', 'ybr') to float (coordinate value)
        Also ensures that object_ids are all unique from those used in actuals data

        :param predictions: predicted labels and bounding box measurements in the
                            standard output format of whatever model created them
                            For Supervisely outputs, this will be a path to a
                            directory with internal files named by frame number
        :param prediction_format: string identifying the format of the prediction data

        :return: does not return; updates self.predictions
        """
        # TODO why not combine this entire method with _build_actuals?
        # TODO (cont'd) only difference is collecting frame numbers and n_objects seen
        if prediction_format == 'maskrcnn':
            for frame_number in self.frame_numbers:
                # hotfix for certain results saved with str instead of float frame numbers
                if isinstance(list(predictions.keys())[0], str):
                    frame_number = str(int(frame_number))
                if frame_number in predictions:
                    bounds = predictions[frame_number]['rois']
                    for i in range(len(bounds)):
                        confidence = float(predictions[frame_number]['scores'][i])
                        if confidence >= self.confidence_threshold:
                            box_data = {
                                'xtl': float(bounds[i][1]),
                                'ytl': float(bounds[i][0]),
                                'xbr': float(bounds[i][3]),
                                'ybr': float(bounds[i][2]),
                                'confidence': confidence
                            }
                            class_name = self.class_names[predictions[frame_number]['class_ids'][i]]
                            if (float(frame_number), class_name) not in self.predictions:
                                self.predictions[(float(frame_number), class_name)] = {}
                            self.predictions[(float(frame_number), class_name)][self._get_next_object_id()] = deepcopy(box_data)
                else:
                    print("frame {} not in predictions".format(frame_number))

        elif prediction_format == 'supervisely':
            # TODO the below assumes that filenames will always have the naming structure 'frame_XXXXX.png.json'
            # TODO (cont'd) update to use regex
            for file_name in os.listdir(predictions):
                frame_id = file_name[6:11]
                frame_number = float(frame_id)
                if frame_number in self.frame_numbers:
                    frame_preds = json.load(open(os.path.join(predictions, "frame_" + frame_id + ".png.json")))
                    for object in frame_preds['objects']:
                        confidence = float(object['tags'][0]['value'])
                        if confidence > self.confidence_threshold:
                            box_data = {
                                'xtl': float(object['points']['exterior'][0][0]),
                                'ytl': float(object['points']['exterior'][0][1]),
                                'xbr': float(object['points']['exterior'][1][0]),
                                'ybr': float(object['points']['exterior'][1][1]),
                                'confidence': confidence
                            }
                            class_name = object['classTitle'].lower()
                            ##### hotfix for data from Summer 2019 and prior
                            if class_name in ['bus', 'car', 'truck', 'vehicle']:
                                class_name = 'vehicle'
                            else:
                                class_name = 'pedestrian'
                            ##### hotfix end
                            if (frame_number, class_name) not in self.predictions:
                                self.predictions[(frame_number, class_name)] = {}
                            self.predictions[(frame_number, class_name)][self._get_next_object_id()] = deepcopy(box_data)

        elif prediction_format == 'zklab_measurement':
            for (frame_index, class_name), objects in predictions.items():
                self.predictions[(frame_index, class_name)] = {}
                for _, object_data in objects.items():
                    if object_data['confidence'] >= self.confidence_threshold:
                        self.predictions[(frame_index, class_name)][self._get_next_object_id()] = object_data

        elif prediction_format == 'tabular':
            for idx in predictions.index:
                frame_number = float(predictions.loc[idx]['frame_number'])
                class_name = predictions.loc[idx]['class_name']
                ##### ##### hotfix for data from Summer 2019 and prior
                if class_name in ['bus', 'car', 'truck', 'vehicle']:
                    class_name = 'vehicle'
                else:
                    class_name = 'pedestrian'
                ##### hotfix end
                confidence = float(predictions.loc[idx]['confidence'])
                if confidence > self.confidence_threshold:
                    box_data = {
                        'xtl': float(predictions.loc[idx]['xtl']),
                        'ytl': float(predictions.loc[idx]['ytl']),
                        'xbr': float(predictions.loc[idx]['xbr']),
                        'ybr': float(predictions.loc[idx]['ybr']),
                        'confidence': confidence
                    }
                    if (frame_number, class_name) not in self.predictions:
                        self.predictions[(frame_number, class_name)] = {}
                    self.predictions[(frame_number, class_name)][self._get_next_object_id()] = deepcopy(box_data)
        else:
            raise NotImplementedError

    def _compute_iou(self, object_a, object_b):
        """Compute the Jaccard index, defined as intersection / union (also called IoU),
        i.e. the percentage of all of the space taken up by the 2 boxes that is overlapping

        :param object_a: dict of object data including 'xtl', 'ytl', 'xbr', 'ybr' (float)
        :param object_b: dict of object data including 'xtl', 'ytl', 'xbr', 'ybr' (float)
        :return: float
        """
        # intersection area
        x_overlap = min(object_a['xbr'], object_b['xbr']) - max(object_a['xtl'], object_b['xtl'])
        y_overlap = min(object_a['ybr'], object_b['ybr']) - max(object_a['ytl'], object_b['ytl'])
        intersection = max(x_overlap, 0) * max(y_overlap, 0)

        if intersection == 0:
            return 0

        # sum of the areas of both boxes
        a_area = (object_a['xbr'] - object_a['xtl']) * (object_a['ybr'] - object_a['ytl'])
        b_area = (object_b['xbr'] - object_b['xtl']) * (object_b['ybr'] - object_b['ytl'])

        return intersection / float(a_area + b_area - intersection)

    def _match_objects(self, iou_threshold):
        """For each frame and each class type, use IoU to minimize the total
        object-hypothesis distance error and thereby determine the true positives
        # TODO this is consistent w/ our MOT evaluation, but different from COCO method:
        # TODO (cont'd) https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
        # TODO (cont'd) Zhengye is implementing greedy matching for multicut so that will be similar to COCO, we
        # TODO (cont'd) should give it a try in here to confirm how different it is from global cost minimization

        :param iou_threshold: float value denoting minimum IoU threshold required to allow for a TP identification
        :return: does not return; updates class matches, total_matches, legal_matches and results_lists attributes
                 to track matched and legal-but-unmatched objects
        """
        matches = {}  # dict of pred to actual
        distances = {}  # dict of frame number to 2d array of pairwise distances
        self.best_matches[iou_threshold] = {}  # dict of pairwise IoUs
        self.legal_matches[iou_threshold] = {}  # dict of pairwise IoUs
        iou_match_lists = []
        iou_legal_match_lists = []

        for frame_class in self.actuals:
            if frame_class in self.predictions:
                actuals = self.actuals[frame_class] # All gt boxes with the same class in the same frame
                preds = self.predictions[frame_class] # All predicted boxes with the same class in the same frame

                # build 2d matrix of distances between predicted and actual objects
                distances[frame_class] = []
                for a in actuals:
                    distance_list = []
                    for b in preds:
                        iou = self._compute_iou(actuals[a], preds[b])
                        if iou >= iou_threshold:  #legal match: a pair of boxes with correct class and IoU higher than threshold
                            distance = (1 - iou)
                            if frame_class not in self.legal_matches[iou_threshold]:
                                self.legal_matches[iou_threshold][frame_class] = {}
                            self.legal_matches[iou_threshold][frame_class][(a, b)] = iou
                            iou_legal_match_lists.append([iou_threshold, frame_class[0], frame_class[1], a, b,
                                                          preds[b]['confidence'], 1, 0, 1, 0, iou])
                        else:
                            distance = self.default_high

                        distance_list.append(distance)
                    distances[frame_class].append(distance_list)

                # minimize total object-hypothesis distance error for available actual and predicted objects
                if distances[frame_class]:
                    row_indices, col_indices = linear_sum_assignment(distances[frame_class])

                    # assign matches
                    indices = list(zip(list(row_indices), list(col_indices)))
                    for row, column in indices:
                        actual = list(actuals.keys())[row]
                        pred = list(preds.keys())[column]
                        matches[pred] = actual
                        iou = 1 - distances[frame_class][row][column]
                        if iou >= iou_threshold:
                            if frame_class not in self.best_matches[iou_threshold]:
                                self.best_matches[iou_threshold][frame_class] = {}
                            self.best_matches[iou_threshold][frame_class][(actual, pred)] = iou
                            iou_match_lists.append([iou_threshold, frame_class[0], frame_class[1], actual, pred,
                                                    preds[pred]['confidence'], 1, 1, 0, 0, iou])


        #match: IoU >= iouthreshold & IoU max & correct class
        initial_results_df = pd.DataFrame(iou_match_lists, columns=['iou_threshold', 'frame_number', 'class_name',
                                                                    'actual', 'detection', 'confidence',
                                                                    'legal_detection', 'TP', 'FP', 'FN', 'iou'])

        #legal IoU >= iouthreshold & correct class
        legal_matches_df = pd.DataFrame(iou_legal_match_lists, columns=['iou_threshold', 'frame_number',
                                                                        'class_name', 'actual', 'detection',
                                                                        'confidence', 'legal_detection', 'TP',
                                                                        'FP', 'FN', 'iou'])

        def _gen_unique(df):
            return df[['iou_threshold', 'frame_number', 'class_name', 'actual', 'detection']].apply(
                lambda x: ''.join(str(x.values)), axis=1)

        initial_results_df['unique'] = _gen_unique(initial_results_df)
        legal_matches_df['unique'] = _gen_unique(legal_matches_df)

        initial_results_df.set_index('unique', inplace=True)
        legal_matches_df.set_index('unique', inplace=True)

        for unique in legal_matches_df.index:
            if unique not in initial_results_df.index:
                initial_results_df = initial_results_df.append(legal_matches_df.loc[unique])
        initial_results_df.reset_index(inplace=True, drop=True)

        # all TP and FP whose IoU >= threshold and class is correct
        return initial_results_df

    def _compute_errors(self, iou_threshold, results_df):
        """Count false positives and false negatives (misses)
        - any actual not matched is missed
        - any prediction not matched is a false positive

        :param iou_threshold: float value denoting minimum IoU threshold required to allow for a TP identification
        :return: does not return; stores lists of object_id ints in class attributes
        """
        # TODO inefficiently written and likely could be combined w/ self._match_objects()

        self.misses[iou_threshold] = {}
        self.false_positives[iou_threshold] = {}

        for frame_class in set(self.actuals.keys()).union(set(self.predictions.keys())):
            self.misses[iou_threshold][frame_class] = []
            self.false_positives[iou_threshold][frame_class] = []

            frame = frame_class[0]

            # loop over all object ids that we need to consider and
            # determine which of the buckets each falls into
            actuals = self.actuals[frame_class] if frame_class in self.actuals else {}
            preds = self.predictions[frame_class] if frame_class in self.predictions else {}

            # false negative if an actual is unmatched
            for a in actuals:
                if (frame_class not in self.best_matches[iou_threshold]) or \
                        (a not in [x[0] for x in self.best_matches[iou_threshold][frame_class]]):
                    self.misses[iou_threshold][frame_class].append(a)
                    results_df = results_df.append(pd.DataFrame({
                        'iou_threshold': iou_threshold,
                        'frame_number': frame,
                        'class_name': frame_class[1],
                        'actual': a,
                        'detection': np.nan,
                        'confidence': np.nan,
                        'legal_detection': 0,
                        'TP': 0,
                        'FP': 0,
                        'FN': 1,
                        'iou': np.nan
                    }, index=[0])).reset_index(drop=True)

            # false positive if not a legal match to any actual
            for b in preds:
                if (frame_class not in self.best_matches[iou_threshold]) or \
                        (b not in [x[1] for x in self.best_matches[iou_threshold][frame_class]]):
                    self.false_positives[iou_threshold][frame_class].append(b)
                    results_df = results_df.append(pd.DataFrame({
                        'iou_threshold': iou_threshold,
                        'frame_number': frame,
                        'class_name': frame_class[1],
                        'actual': np.nan,
                        'detection': b,
                        'confidence': preds[b]['confidence'],
                        'legal_detection': 0,
                        'TP': 0,
                        'FP': 1,
                        'FN': 0,
                        'iou': np.nan
                    }, index=[0])).reset_index(drop=True)

        return results_df

    def _get_matches_and_errors(self):
        """Compute all matches and errors across all iou thresholds

        :return: does not return; updates self.results_df
        """
        for iou_threshold in self.iou_thresholds:
            print("generating results at IoU threshold {}".format(iou_threshold))
            initial_results_df = self._match_objects(iou_threshold)
            iou_threshold_results_df = self._compute_errors(iou_threshold, initial_results_df).drop_duplicates()
            self.results_df = self.results_df.append(iou_threshold_results_df)

    def get_n_objects(self, frame_number=None):
        """Return the number of ground truth objects, at either a given time or over all frames

        :param frame_number: None or float, frame ID number
        :return: dict of string (class_name) to int (number of objects in the video or frame)
        """
        actuals_df = deepcopy(self.results_df[~np.isnan(self.results_df['actual'])])[[
            'class_name', 'frame_number', 'actual']].drop_duplicates()

        return_dict = {}
        total_n = 0
        if frame_number is not None:
            for class_name in self.class_names[1:]:  # skip 'bg'
                class_n = len(actuals_df[(actuals_df['frame_number'] == frame_number) &
                                         (actuals_df['class_name'] == class_name)])
                return_dict[class_name] = class_n
                total_n += class_n

        else:
            for class_name in self.class_names[1:]:  # skip 'bg'
                class_n = len(actuals_df[(actuals_df['class_name'] == class_name)])
                return_dict[class_name] = class_n
                total_n += class_n

        return_dict['total'] = total_n
        return return_dict

    def get_n_detections(self, frame_number=None):
        """Return the number of model detections measured (regardless of correctness)
        at either a given time or over all frames

        :param frame_number: None or float, frame ID number
        :return: dict of string (class_name) to int (number of objects in the video or frame)
        """
        detections_df = deepcopy(self.results_df[~np.isnan(self.results_df['detection'])])[
            ['class_name', 'frame_number', 'detection']].drop_duplicates()

        return_dict = {}
        total_n = 0
        if frame_number is not None:
            for class_name in self.class_names[1:]:  # skip 'bg'
                class_n = len(detections_df[
                                  (detections_df['frame_number'] == frame_number) &
                                  (detections_df['class_name'] == class_name)
                                  ])
                return_dict[class_name] = class_n
                total_n += class_n

        else:
            for class_name in self.class_names[1:]:  # skip 'bg'
                class_n = len(detections_df[
                                  (detections_df['class_name'] == class_name)
                              ])
                return_dict[class_name] = class_n
                total_n += class_n

        return_dict['total'] = total_n
        return return_dict

    def get_n_legally_matched_detections(self, iou_threshold, frame_number=None):
        """Find the count of detections legally matched at a given IoU threshold
        Optionally for a single frame (defaults to computing across all frames)

        :param iou_threshold: float
        :param frame_number: None or float, frame ID number
        :return: dict of string (class name) to float (N objects)
        """
        frame_numbers = [frame_number] if frame_number is not None else self.frame_numbers
        matches = {}
        total_count = 0
        for class_name in self.class_names[1:]:  # skip 'bg'
            match_count = 0
            for frame_number in frame_numbers:
                frame_class = (frame_number, class_name)
                if frame_class in self.legal_matches[iou_threshold]:
                    # the same detection could be matched to multiple actuals, so ensure we count unique detections, not matches
                    match_count += len(set([x[1] for x in self.legal_matches[iou_threshold][frame_class].keys()]))
            matches[class_name] = match_count
            total_count += match_count
        matches['total'] = total_count
        return matches

    def get_n_matched_actuals(self, iou_threshold, frame_number=None):
        """Find the count of ground truth objects matched at a given IoU threshold
        Optionally for a single frame (defaults to computing across all frames)

        :param iou_threshold: float
        :param frame_number: None or float, frame ID number
        :return: dict of string (class name) to float (N objects)
        """
        frame_numbers = [frame_number] if frame_number is not None else self.frame_numbers
        matches = {}
        total_count = 0
        for class_name in self.class_names[1:]:  # skip 'bg'
            match_count = 0
            for frame_number in frame_numbers:
                frame_class = (frame_number, class_name)
                if frame_class in self.best_matches[iou_threshold]:
                    match_count += len(self.best_matches[iou_threshold][frame_class])
            matches[class_name] = match_count
            total_count += match_count
        matches['total'] = total_count
        return matches

    def get_n_misses(self, iou_threshold, frame_number=None):
        """Return the count of objects missed by the tracker, at either a given time or over all frames

        :param iou_threshold: float
        :param frame_number: None or float, frame ID number
        :return: dict of string (class_name) to float
        """
        frame_numbers = [frame_number] if frame_number is not None else self.frame_numbers
        misses = {}
        total_count = 0
        for class_name in self.class_names[1:]:  # skip 'bg'
            miss_count = 0
            for frame_number in frame_numbers:
                frame_class = (frame_number, class_name)
                if frame_class in self.misses[iou_threshold]:
                    miss_count += len(self.misses[iou_threshold][frame_class])
            misses[class_name] = miss_count
            total_count += miss_count
        misses['total'] = total_count
        return misses

    def get_n_false_positives(self, iou_threshold, frame_number=None):
        """Return the count of objects mistakenly identified by the tracker
        that do not actually exist at all, at either a given time or over all frames

        :param iou_threshold: float
        :param frame_number: None or float, frame ID number
        :return: dict of string (class_name) to float
        """
        frame_numbers = [frame_number] if frame_number is not None else self.frame_numbers
        false_positives = {}
        total_count = 0
        for class_name in self.class_names[1:]:  # skip 'bg'
            fp_count = 0
            for frame_number in frame_numbers:
                frame_class = (frame_number, class_name)
                if frame_class in self.false_positives[iou_threshold]:
                    fp_count += len(self.false_positives[iou_threshold][frame_class])
            false_positives[class_name] = fp_count
            total_count += fp_count
        false_positives['total'] = total_count
        return false_positives

    def _compute_recall(self, df, iou_threshold, class_name, frame_number=None):
        """Recall is the proportion of TP objects out of the possible positives
        As we work down a df ordered by confidence (desc.), recall increases monotonically

        :param df: pd.DataFrame
        :param iou_threshold: float
        :param frame_number: float
        :param class_name: str
        :return: list of float
        """
        n_actuals = self.get_n_objects(frame_number=frame_number)[class_name]
        df = df[df['iou_threshold'] == iou_threshold]
        df = df.sort_values(by='confidence', ascending=False)
        recalls = []
        matched_boxes = set()

        for i, row in df.iterrows():
            if row['legal_detection'] == 1:
                matched_boxes.add(row['actual'])
            potential_true_positives = len(matched_boxes)
            recall = (potential_true_positives / n_actuals) if n_actuals else 0.0  # TODO 1 or 0 or np.nan?
            recalls.append(recall)

        df['recall'] = np.asarray(recalls)
        return df

    @staticmethod
    def _smooth_precisions(precision_list):
        """Ensure monotonicity

        :param precision_list: list of float
        :return precision_list: list of float
        """
        precision_list.reverse()
        max_number = 0
        for i in range(len(precision_list)):
            if i == 0 and np.isnan(precision_list[i]):
                pass
            elif np.isnan(precision_list[i - 1]) and np.isnan(precision_list[i]):
                pass
            else:
                precision_list[i] = max_number = max(max_number, precision_list[i]) if not np.isnan(precision_list[i]) else max_number

        precision_list.reverse()
        return precision_list

    def _get_precision_list(self, class_name, recall_levels=None, iou_threshold=None, frame_number=None, smooth=True):
        """Calculate precision for each recall level

        :param class_name: string name of object class
        :param recall_levels: list of float, each a recall level at which to measure precision; if None, defaults to self.recall_levels
        :param iou_threshold: float value of minimum valid IoU; if None, defaults to self.primary_iou_threshold
                              must be one of the values passed to the constructor (either for iou_thresholds or
                              primary_iou_threshold) or else matches and errors will not be available
        :param frame_number: None or float, frame ID number
        :param smooth: ensure precision decreases monotonically as recall increases
        :return precision_list: list of float
        """
        recall_levels = recall_levels if recall_levels else self.recall_levels
        iou_threshold = iou_threshold if iou_threshold is not None else self.primary_iou_threshold
        if iou_threshold not in self.iou_thresholds:
            raise ValueError("iou_threshold must be one of the values passed into the class constructor")
        results = deepcopy(self.results_df)
        results = results.loc[(results['class_name'] == class_name) & (results['iou_threshold'] == iou_threshold)]

        if frame_number is not None:
            results = results.loc[results['frame_number'] == frame_number]

        # rank by confidence and compute recall moving downward
        if results.empty:
            return [np.nan] * len(recall_levels)
        else:
            detections = results[results['detection'].notnull()]
            detections = self._compute_recall(detections, iou_threshold, class_name, frame_number)

            if len(detections) == 0:
                return [np.nan] * len(recall_levels)

            # # compute average precision across recall levels
            precision_list = []
            for recall in recall_levels:
                # to compute precision, there must exist results at or below the given recall value
                if min(detections['recall']) > recall:
                    precision_list.append(1.0)  # TODO or np.nan ?
                # TODO to compute precision, shouldn't it be true that there must exist results at least as high as the given recall value?
                elif max(detections['recall']) < recall:
                    precision_list.append(0.0)  # TODO or np.nan ?
                else:
                    precision = sum(detections.loc[(detections['recall'] <= recall)
                                                   & ~pd.isnull(detections['TP'])]['TP']) / \
                                len(detections.loc[(detections['recall'] <= recall)])
                    precision_list.append(precision)

            if smooth:
                precision_list = self._smooth_precisions(precision_list)

            assert len(precision_list) == len(recall_levels)

            # if first recall value is 0.0, first precision value should be 1.0
            if recall_levels[0] == 0:
                precision_list[0] = 1.0

            return precision_list

    def plot_precision_recall_curve(self, class_name, recall_levels=None, iou_threshold=None, frame_number=None,
                                    smooth=True):
        """Save image of precision-recall curve for a given class

        :param class_name: string name of object class
        :param recall_levels: list of float, each a recall level at which to measure precision; if None, defaults to self.recall_levels
        :param iou_threshold: float value of minimum valid IoU; if None, defaults to self.primary_iou_threshold
                              must be one of the values passed to the constructor (either for iou_thresholds or
                              primary_iou_threshold) or else matches and errors will not be available
        :param frame_number: None or float, frame ID number
        :param smooth: ensure precision decreases monotonically as recall increases
        :return: does not return; saves .png to /test_data
        """
        recall_levels = recall_levels if recall_levels else self.recall_levels
        iou_threshold = iou_threshold if iou_threshold is not None else self.primary_iou_threshold
        if iou_threshold not in self.iou_thresholds:
            raise ValueError("iou_threshold must be one of the values passed into the class constructor")

        precision_list = self._get_precision_list(class_name, recall_levels, iou_threshold, frame_number, smooth)
        plt.plot(recall_levels, precision_list, marker='o')
        ax = plt.gca()
        ax.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.01))
        plt.xlabel('Recall (TP / all objects)')
        plt.ylabel('Precision (TP / all detections)')
        plt.grid(color='lightgrey', linestyle='--', linewidth=0.5)

        plot_name = 'test_data/PRcurve_class_{}_iouthr_{}'.format(class_name, iou_threshold)
        if frame_number is not None:
            plot_name += '_frame_{}'.format(frame_number)
        if smooth:
            plot_name += '_smoothed'

        plt.savefig(plot_name + '.png')
        plt.close()

    def plot_all_precision_recall_curves(self, class_name, recall_levels=None, frame_number=None, smooth=True):
        """Save image of all precision-recall curves (one per IOU threshold) for a given class

        :param class_name: string name of object class
        :param recall_levels: list of float, each a recall level at which to measure precision; if None, defaults to self.recall_levels
        :param iou_threshold: float value of minimum valid IoU; if None, defaults to self.primary_iou_threshold
                              must be one of the values passed to the constructor (either for iou_thresholds or
                              primary_iou_threshold) or else matches and errors will not be available
        :param frame_number: None or float, frame ID number
        :param smooth: ensure precision decreases monotonically as recall increases
        :return: does not return; saves .png to /test_data
        """
        recall_levels = recall_levels if recall_levels else self.recall_levels

        plot_name = 'test_data/PRcurves_class_{}'.format(class_name)
        if frame_number is not None:
            plot_name += '_frame_{}'.format(frame_number)
        if smooth:
            plot_name += '_smoothed'

        for iou_threshold in self.iou_thresholds:
            precision_list = self._get_precision_list(class_name, recall_levels, iou_threshold, frame_number, smooth)
            plt.plot(recall_levels, precision_list, marker='o', label='iou_threshold: {}'.format(iou_threshold))

        ax = plt.gca()
        ax.set(xlim=(-0.01, 1.7), ylim=(-0.01, 1.01))
        plt.xlabel('Recall (TP / all objects)')
        plt.ylabel('Precision (TP / all detections)')
        plt.grid(color='lightgrey', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.savefig(plot_name + '.png')
        plt.close()

    def generate_video(self,calibrated_video,result_path):
        '''Generate prediction video
        :param: calibrated_video: calibrated video file path(String)
        :param: result_path: path to save the video(String)
        '''
        predictions = self.predictions
        vidcap = cv2.VideoCapture(calibrated_video)
        success, image = vidcap.read()
        count = 0
        height, width, layers = image.shape
        video = cv2.VideoWriter(result_path, cv2.VideoWriter.fourcc(*'H264'), 10, (width, height))
        while success:
            try:
                for i in predictions[(float(count),'vehicle')]:
                    start_point = (int(predictions[(float(count),'vehicle')][i]['xtl']),int(predictions[(float(count),'vehicle')][i]['ytl']))
                    end_point = (int(predictions[(float(count),'vehicle')][i]['xbr']),int(predictions[(float(count),'vehicle')][i]['ybr']))
                    color= (0,255,0)
                    thickness = 2
                    image = cv2.rectangle(image, start_point, end_point, color, thickness)
                for i in predictions[(float(count),'pedestrian')]:
                    start_point = (int(predictions[(float(count),'pedestrian')][i]['xtl']),int(predictions[(float(count),'pedestrian')][i]['ytl']))
                    end_point = (int(predictions[(float(count),'pedestrian')][i]['xbr']),int(predictions[(float(count),'pedestrian')][i]['ybr']))
                    color= (255,0,0)
                    thickness = 2
                    image = cv2.rectangle(image, start_point, end_point, color, thickness)
                video.write(image)
                success, image = vidcap.read()
                count += 1
            except KeyError:
                success, image = vidcap.read()
                print('Read a new frame: ', success)
                count += 1
        print('video contains:',count)
        cv2.destroyAllWindows()

        video.release()

    def get_average_precision(self, class_name, recall_levels=None, iou_threshold=None, frame_number=None, smooth=True):
        """Calculate the average precision (true positives divided by all predictions)
        for a given object class, using a given iou_threshold, across a set of recall levels
        Optionally for a single frame (defaults to computing across all frames)

        :param class_name: string name of object class
        :param recall_levels: list of float, each a recall level at which to measure precision; if None, defaults to self.recall_levels
        :param iou_threshold: float value of minimum valid IoU; if None, defaults to self.primary_iou_threshold
                              must be one of the values passed to the constructor (either for iou_thresholds or
                              primary_iou_threshold) or else matches and errors will not be available
        :param frame_number: None or float, frame ID number
        :param smooth: ensure precision decreases monotonically as recall increases
        :return: float
        """
        precision_list = self._get_precision_list(class_name, recall_levels, iou_threshold, frame_number, smooth)
        precision_list = [x for x in precision_list if ~np.isnan(x)]
        return sum(precision_list) / len(precision_list) if len(precision_list) else 0  # TODO 0 or np.nan ?

    def get_all_average_precisions(self, recall_levels=None, iou_thresholds=None, frame_number=None, smooth=True):
        """Compute the average precision for each class and across
        all classes, for all IoU thresholds for which we have data
        Optionally for a single frame (defaults to computing across all frames)

        :param recall_levels: list of float, each a recall level at which to measure precision; if None, defaults to self.recall_levels
        :param iou_thresholds: list of float; if None, defaults to self.iou_thresholds
        :param frame_number: None or float, frame ID number
        :param smooth: ensure precision decreases monotonically as recall increases
        :return: dict of string (class name) to float (avg. precision)
        """
        class_precisions = {}
        total_sum = 0
        total_count = 0
        recall_levels = recall_levels if recall_levels else self.recall_levels
        iou_thresholds = iou_thresholds if iou_thresholds else self.iou_thresholds
        for class_name in self.class_names[1:]:  # skip 'bg'
            precision = 0
            count = 0
            for iou_threshold in iou_thresholds:
                ct_precision = self.get_average_precision(class_name, recall_levels, iou_threshold, frame_number, smooth)
                if not np.isnan(ct_precision):
                    precision += ct_precision
                    count += 1
            class_precisions[class_name] = precision / count if count else 0
            total_sum += precision
            total_count += count
        class_precisions['total'] = total_sum / total_count if total_count else 0
        return class_precisions

    def get_mean_average_precision(self, recall_levels=None, iou_thresholds=None, frame_number=None, smooth=True):
        """Compute the mean of the average precision across all
        classes and all IoU thresholds for which we have data
        Optionally for a single frame (defaults to computing across all frames)

        :param recall_levels: list of float, each a recall level at which to measure precision; if None, defaults to self.recall_levels
        :param iou_thresholds: list of float; if None, defaults to self.iou_thresholds
        :param frame_number: None or float, frame ID number
        :param smooth: ensure precision decreases monotonically as recall increases
        :return: float
        """
        return self.get_all_average_precisions(recall_levels, iou_thresholds, frame_number, smooth)['total']

    def get_best_matches_ious(self, iou_threshold, frame_number=None):
        """Find the avg. IoU for only the best matched detections
        at a given IoU threshold, for each class and across all classes
        Optionally for a single frame (defaults to computing across all frames)

        :param iou_threshold: float
        :return: dict of string (class name) to float (mean IoU)
        """
        frame_numbers = [frame_number] if frame_number is not None else self.frame_numbers
        class_ious = {}
        total_sum = 0
        total_count = 0
        for class_name in self.class_names[1:]:  # skip 'bg'
            iou_sum = 0
            iou_count = 0
            for frame_number in frame_numbers:
                frame_class = (frame_number, class_name)
                if frame_class in self.best_matches[iou_threshold]:
                    for pair, iou in self.best_matches[iou_threshold][frame_class].items():
                        iou_sum += iou
                        iou_count += 1
            class_ious[class_name] = iou_sum / iou_count if iou_count else 0
            total_sum += iou_sum
            total_count += iou_count
        class_ious['total'] = total_sum / total_count if total_count else 0
        return class_ious

    def get_all_legal_matches_ious(self, iou_threshold, frame_number=None):
        """Find the avg. IoU for each legally matched detection
        at a given IoU threshold, for each class and across all classes
        Optionally for a single frame (defaults to computing across all frames)

        :param iou_threshold: float
        :return: dict of string (class name) to float (mean IoU)
        """
        frame_numbers = [frame_number] if frame_number is not None else self.frame_numbers
        class_ious = {}
        total_sum = 0
        total_count = 0
        for class_name in self.class_names[1:]:  # skip 'bg'
            iou_sum = 0
            iou_count = 0
            for frame_number in frame_numbers:
                frame_class = (frame_number, class_name)
                if frame_class in self.legal_matches[iou_threshold]:
                    for pair, iou in self.legal_matches[iou_threshold][frame_class].items():
                        iou_sum += iou
                        iou_count += 1
            class_ious[class_name] = iou_sum / iou_count if iou_count else 0
            total_sum += iou_sum
            total_count += iou_count
        class_ious['total'] = total_sum / total_count if total_count else 0
        return class_ious

    def get_all_metrics(self, iou_threshold=None, frame_number=None, return_format='json'):
        """Return all intermediate and final metrics, at either a given time or over all frames

        :param iou_threshold: float
        :param frame_number: None or float, frame ID number
        :param return_format: String, one of ['json', 'df']
        :return: dict of string (metric name) to float (metric value)
        """
        iou_threshold = iou_threshold if iou_threshold is not None else self.primary_iou_threshold

        return_dict = {
            'actual_boxes': self.get_n_objects(frame_number=frame_number),
            'detections': self.get_n_detections(frame_number=frame_number),
            'matched_actuals': self.get_n_matched_actuals(iou_threshold=iou_threshold, frame_number=frame_number),
            'legally_matched_detections': self.get_n_legally_matched_detections(iou_threshold=iou_threshold, frame_number=frame_number),
            'misses': self.get_n_misses(iou_threshold=iou_threshold, frame_number=frame_number),
            'false_positives': self.get_n_false_positives(iou_threshold=iou_threshold, frame_number=frame_number),
            'best_matches_mIoU': self.get_best_matches_ious(iou_threshold=iou_threshold, frame_number=frame_number),
            'legal_matches_mIoU': self.get_all_legal_matches_ious(iou_threshold=iou_threshold, frame_number=frame_number),
            'AP': self.get_all_average_precisions(iou_thresholds=[iou_threshold], frame_number=frame_number),
            'mAP': self.get_mean_average_precision(frame_number=frame_number)
        }
        return return_dict if return_format == 'json' else pd.DataFrame.from_dict(return_dict)


if __name__=='__main__':
    with open('./traffic_video_GP010589_190720_0310_0440_90sec_calibrated_stable.json', 'r') as f:
        actual = json.load(f)
        #print('the type of actual is: ', type(actual))

    with open('./result_in_json/res_for_eval.json', 'r') as f1:
        pred = json.load(f1)
        #print('the type of pred is: ', type(pred))

   # pred = {float(k): v for k, v in pred.items()}

    # res = DetectionMeasurement(actual, pred, 'maskrcnn',frame_numbers=list(np.linspace(0,900,901)))
    # metrics = res.get_all_metrics()


    res = DetectionMeasurement(actuals=actual, predictions=pred, prediction_format='maskrcnn', recall_levels=np.linspace(0.0, 1.0, 3))
    metrics = res.get_all_metrics()
    print('the type of metrics: ', type(metrics))
    print('metrics: ', metrics)

    with open('./metrics.json', 'w') as fp:
        json.dump(metrics, fp)

    #f = open("auc.metric","w+")
    #f.write(metrics)
    #f.close()
    print('create output metrics.json')

