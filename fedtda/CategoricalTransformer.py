import numpy as np
import pandas as pd
import psutil
from collections import Counter
from scipy.stats import norm

from fedtda.transformer_base import BaseTransformer


class CategoricalTransformer(BaseTransformer):

    INPUT_TYPE = 'categorical'
    OUTPUT_TYPES = {'value': 'float'}
    DETERMINISTIC_REVERSE = True
    COMPOSITION_IS_IDENTITY = True

    mapping = None
    intervals = None
    starts = None
    means = None
    dtype = None
    _get_category_from_index = None

    def __setstate__(self, state):
        """Replace any ``null`` key by the actual ``np.nan`` instance."""
        intervals = state.get('intervals')
        if intervals:
            for key in list(intervals):
                if pd.isna(key):
                    intervals[np.nan] = intervals.pop(key)

        self.__dict__ = state

    def __init__(self, fuzzy=False, clip=False):
        self.fuzzy = fuzzy
        self.clip = clip

    @staticmethod
    def _get_intervals(data_list):
        """Compute intervals for each categorical value.

        Args:
            data (pandas.Series):
                Data to analyze.

        Returns:
            dict:
                intervals for each categorical value (start, end).
        """
        for i in range(len(data_list)):
            data_list[i] = data_list[i].fillna(np.nan)

        # [data.value_counts(dropna=False) for data in data_list]
        frequencies = fed_value_counts(data_list)

        start = 0
        end = 0
        elements = sum([len(data) for data in data_list])

        intervals = {}
        means = []
        starts = []
        for value, frequency in frequencies.items():
            prob = frequency / elements
            end = start + prob
            mean = start + prob / 2
            std = prob / 6
            if pd.isna(value):
                value = np.nan

            intervals[value] = (start, end, mean, std)
            # intervals[value] = (start, end)
            means.append(mean)
            starts.append((value, start))
            start = end

        means = pd.Series(means, index=list(frequencies.keys()))
        starts = pd.DataFrame(starts, columns=['category', 'start']).set_index('start')

        return intervals, means, starts
        # return intervals, starts

    @staticmethod
    def _get_intervals_new(data_list):
        """Compute intervals for each categorical value.

        Args:
            data (pandas.Series):
                Data to analyze.

        Returns:
            dict:
                intervals for each categorical value (start, end).
        """
        for i in range(len(data_list)):
            data_list[i] = data_list[i].fillna(np.nan)

        # [data.value_counts(dropna=False) for data in data_list]
        frequencies = fed_value_counts(data_list)

        start = 0
        end = 0
        elements = sum([len(data) for data in data_list])

        intervals = {}
        means = []
        starts = []
        for value, frequency in frequencies.items():
            prob = frequency / elements
            end = start + prob
            mean = start + prob / 2
            # std = prob / 6
            if pd.isna(value):
                value = np.nan

            # intervals[value] = (start, end, mean, std)
            intervals[value] = (start, end)
            means.append(mean)
            starts.append((value, start))
            start = end

        means = pd.Series(means, index=list(frequencies.keys()))
        starts = pd.DataFrame(starts, columns=['category', 'start']).set_index('start')

        return intervals, means, starts
        # return intervals, starts

    def _fit(self, data_list):
        """Fit the transformer to the data.

        Create the mapping dict to save the label encoding.
        Finally, compute the intervals for each categorical value.

        Args:
            data (pandas.Series):
                Data to fit the transformer to.
        """
        self.mapping = {}
        self.dtype = data_list[0].dtype

        self.intervals, self.means, self.starts = self._get_intervals(data_list)
        # self.intervals, self.means, self.starts = self._get_intervals_new(data_list)
        # self._get_category_from_index = list(self.means.index).__getitem__

    def _transform_by_category_new(self, data_list):
        """Transform the data by iterating over the different categories."""
        # data_num = sum([len(data) for data in data_list])
        # result = np.empty(shape=(data_num, ), dtype=float)

        # norm_data = np.empty(shape=(data_num, ), dtype=float)
        # for i in range(data_num):
        #     norm_data[i] = (i+1) / (data_num+1)
        # norm_data = norm.ppf(norm_data)

        # loop over categories
        result_list = []
        for data in data_list:
            result = np.empty(shape=(len(data), ), dtype=float)
            for category, values in self.intervals.items():
                # mean, std = values[2:]
                start, end = values
                p = end - start
                if category is np.nan:
                    mask = data.isna()
                else:
                    mask = (data.to_numpy() == category)

                # 在start和end之间取等量的分点x,用norm.ppf(来代替)
                # result[mask] = norm.rvs(mean, std, size=mask.sum())
                result[mask] = np.random.rand(mask.sum()) * p + start

            result_list.append(norm.ppf(result))

        return result_list

    def _transform_by_category(self, data_list):
        """Transform the data by iterating over the different categories."""
        # data_num = sum([len(data) for data in data_list])
        # result = np.empty(shape=(data_num, ), dtype=float)

        # norm_data = np.empty(shape=(data_num, ), dtype=float)
        # for i in range(data_num):
        #     norm_data[i] = (i+1) / (data_num+1)
        # norm_data = norm.ppf(norm_data)

        # loop over categories
        result_list = []
        for data in data_list:
            result = np.empty(shape=(len(data), ), dtype=float)
            for category, values in self.intervals.items():
                mean, std = values[2:]
                start, end = values[:2]
                # p = end - start
                if category is np.nan:
                    mask = data.isna()
                else:
                    mask = (data.to_numpy() == category)

                # 在start和end之间取等量的分点x,用norm.ppf(来代替)
                result[mask] = norm.rvs(mean, std, size=mask.sum())
                # result[mask] = np.array((end-start)/2)

            # result_list.append(norm.ppf(result))
            result_list.append(result)

        return result_list


    def _transform_by_row(self, data):
        """Transform the data row by row."""
        return data.fillna(np.nan).apply(self._get_value).to_numpy()

    def _transform(self, data_list):
        """Transform categorical values to float values.

        Replace the categories with their float representative value.

        Args:
            data (pandas.Series):
                Data to transform.

        Returns:
            numpy.ndarray:
        """
        # if len(self.intervals) < len(data):
        # return self._transform_by_category_new(data_list)
        return self._transform_by_category(data_list)

        # return self._transform_by_row(data)

    def _get_value(self, category):
        """Get the value that represents this category."""
        if pd.isna(category):
            category = np.nan

        mean, std = self.intervals[category][2:]

        if self.fuzzy:
            return norm.rvs(mean, std)

        return mean

    def _reverse_transform_by_matrix(self, data):
        """Reverse transform the data with matrix operations."""
        num_rows = len(data)
        num_categories = len(self.means)

        data = np.broadcast_to(data, (num_categories, num_rows)).T
        means = np.broadcast_to(self.means, (num_rows, num_categories))
        diffs = np.abs(data - means)
        indexes = np.argmin(diffs, axis=1)

        self._get_category_from_index = list(self.means.index).__getitem__
        return pd.Series(indexes).apply(self._get_category_from_index).astype(self.dtype)

    def _reverse_transform_by_category_new(self, data):
        """Reverse transform the data by iterating over all the categories."""
        result = np.empty(shape=(len(data), ), dtype=self.dtype)

        # loop over categories
        for category, values in self.intervals.items():
            start = values[0]
            mask = (start <= data.to_numpy())
            result[mask] = category

        return pd.Series(result, index=data.index, dtype=self.dtype)

    def _reverse_transform_by_category(self, data):
        """Reverse transform the data by iterating over all the categories."""
        result = np.empty(shape=(len(data), ), dtype=self.dtype)

        # loop over categories
        for category, values in self.intervals.items():
            start = values[0]
            mask = (start <= data.to_numpy())
            result[mask] = category

        return pd.Series(result, index=data.index, dtype=self.dtype)

    def _get_category_from_start(self, value):
        lower = self.starts.loc[:value]
        return lower.iloc[-1].category

    def _reverse_transform_by_row(self, data):
        """Reverse transform the data by iterating over each row."""
        return data.apply(self._get_category_from_start).astype(self.dtype)

    def _reverse_transform(self, data):
        """Convert float values back to the original categorical values.

        Args:
            data (pd.Series):
                Data to revert.

        Returns:
            pandas.Series
        """
        data = self._normalize(data)

        num_rows = len(data)
        num_categories = len(self.intervals)

        # total shape * float size * number of matrices needed
        needed_memory = num_rows * num_categories * 8 * 3
        available_memory = psutil.virtual_memory().available
        if available_memory > needed_memory:
            return self._reverse_transform_by_matrix(data)

        if num_rows > num_categories:
            return self._reverse_transform_by_category(data)

        # loop over rows
        return self._reverse_transform_by_row(data)


def fed_value_counts(data_list):
    """
    统计离散列全局各值出现次数
    :param data_list:
    :return:
    """
    frequencies_list = [data.value_counts(dropna=False) for data in data_list]
    head_set_list = [set(frequencies.index) for frequencies in frequencies_list]

    head_set = set()
    for i in range(len(head_set_list)):
        head_set = head_set.union(head_set_list[i])

    categories = list(head_set)
    frequencies = pd.Series(0, index=categories)

    for category in categories:
        for i in range(len(frequencies_list)):
            if category in frequencies_list[i].index:
                frequencies[category] += frequencies_list[i][category]

    return frequencies
