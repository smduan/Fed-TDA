from fedtda.utils import get_instance, random_state, validate_random_state
from fedtda.univariate import Univariate

import logging
import sys
import warnings
import pandas as pd
import numpy as np
from scipy import stats
DEFAULT_DISTRIBUTION = Univariate
LOGGER = logging.getLogger(__name__)
EPSILON = np.finfo(np.float32).eps


class Multivariate:
    """

    """
    covariance = None
    columns = None
    univariates = None

    def __init__(self, distribution=None, random_state=None):
        self.random_state = validate_random_state(random_state)
        self.distribution = distribution
        self.fitted = False

    def __repr__(self):
        """Produce printable representation of the object."""
        if self.distribution == DEFAULT_DISTRIBUTION:
            distribution = ''
        elif isinstance(self.distribution, type):
            distribution = f'distribution="{self.distribution.__name__}"'
        else:
            distribution = f'distribution="{self.distribution}"'

        return f'GaussianMultivariate({distribution})'

    def sget_covariance(self, x_list):
        """
        Compute covariance matrix with transformed data.
        :param x_list:
        :return: (numpy.ndarray)computed covariance matrix.
        """

    def fit(self, x_list):
        """

        :param x_list:
        :return:
        """
        LOGGER.info('Fitting %s', self)

        clients_num = len(x_list)
        for x in x_list:
            if not isinstance(x, pd.DataFrame):
                x = pd.DataFrame(x)

        columns = []
        univariates = []

        for column_name, _ in x_list[0].items():
            if isinstance(self.distribution, dict):
                # distribution = self.distribution.get(column_name, DEFAULT_DISTRIBUTION)
                distribution = self.distribution[column_name.split('.')[0]]
            else:
                distribution = self.distribution

            LOGGER.debug('Fitting column %s to %s', column_name, distribution)

            univariate = Univariate()
            try:
                # columns_list = []
                # for i in range(clients_num):
                #     column = x_list[i][column_name]
                #     columns_list.append(column)
                univariate.fed_fit(x_list, column_name, distribution)
            except RuntimeError:
                warning_message = (
                    f'Unable to fit to a {distribution} distribution for column {column_name}. '
                )
                warnings.warn(warning_message)

            columns.append(column_name)
            univariates.append(univariate)

        self.columns = columns
        self.univariates = univariates

        LOGGER.debug('Computing covariance')
        self.covariance = self.get_covariance(x_list)
        # self.covariance.to_csv("covariance_example.csv")
        self.fitted = True

        LOGGER.debug('GaussianMultivariate fitted successfully')

    def get_covariance(self, x_list):
        """
        Compute covariance matrix with transformed data.

        Args:
            x_list (numpy.ndarray):
                Data for which the covariance needs to be computed.

        Returns:
            numpy.ndarray:
                computed covariance matrix.
        """
        result_list = self.transform_to_normal(x_list)

        # result = np.concatenate(result_list)
        # covariance = pd.DataFrame(data=result).corr().to_numpy()
        # covariance = np.nan_to_num(covariance, nan=0.0)

        part_covariance_list = []
        num_all = 0
        m = result_list[0].shape[1]

        columns_sum = np.zeros(m)
        for i in range(m):
            for result in result_list:
                columns_sum += result.sum(axis=0)
                if i == 0:
                    num_all += len(result)
        columns_mean = columns_sum / num_all

        for result in result_list:
            part_covariance = np.zeros((m,m))
            for i in range(m):
                for j in range(i,m):
                    part_covariance[i, j] = ((result[:,i]-columns_mean[i]) * (result[:,j]-columns_mean[j])).sum()
                    part_covariance[j, i] = part_covariance[i, j]
            part_covariance = np.nan_to_num(part_covariance, nan=0.0)
            part_covariance_list.append(part_covariance)

        covariance = np.sum(np.stack(part_covariance_list), axis=0) / num_all

        # If singular, add some noise to the diagonal
        if np.linalg.cond(covariance) > 1.0 / sys.float_info.epsilon:
            covariance = covariance + np.identity(covariance.shape[0]) * EPSILON

        self.new_columns = []
        for column_name, univariate in zip(self.columns, self.univariates):
            if len(univariate._params) == 2:
                self.new_columns.append(column_name)
            else:
                self.new_columns += univariate.new_column_name
        return pd.DataFrame(covariance, index=self.new_columns, columns=self.new_columns)
        # return pd.DataFrame(covariance)

    def transform_to_normal(self, x_list):
        """
        暂时不采用这种cdf后ppf的做法，而是直接采用(x-loc)/scale
        :param x_list:
        :return:
        """
        # cdf_x = []
        # for column_name, univariate in zip(self.columns, self.univariates):
        #     if column_name in X:
        #         column = X[column_name]
        #         cdf_x.append(univariate.cdf(column.to_numpy()).clip(EPSILON, 1 - EPSILON))
        #
        # return stats.norm.ppf(np.column_stack(cdf_x))
        norm_x_list = []
        start = 0
        for x in x_list:
            norm_x = []
            for column_name, univariate in zip(self.columns, self.univariates):
                if column_name in x:
                    column = x[column_name]
                    norm_x.append(univariate.norm(column, start))
            start += len(x)
            norm_x_list.append(np.column_stack(norm_x))

        return norm_x_list
