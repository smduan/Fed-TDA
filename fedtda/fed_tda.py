import warnings
import logging
import pandas as pd
import numpy as np
import os
import math
import uuid
from copy import deepcopy
from scipy import stats

from fedtda.table import Table
from fedtda.multivariate import Multivariate
from fedtda.utils import validate_random_state, progress_bar_wrapper, handle_sampling_error, ConstraintsNotMetError, devide_list

LOGGER = logging.getLogger(__name__)
FIXED_RNG_SEED = 73251
TMP_FILE_NAME = '.sample.csv.temp'
COND_IDX = str(uuid.uuid4())
EPSILON = np.finfo(np.float32).eps

class FedTabularDataSyn:
    """

    """
    _model = None
    _metadata = None
    _DTYPE_TRANSFORMERS = {
        'i': 'integer',
        'f': 'float',
        'O': 'one_hot_encoding',
        'b': 'boolean',
        'M': 'datetime',
    }

    def __init__(self, field_names=None, field_types=None, field_transformers=None,
                 primary_key=None, constraints=None, table_metadata=None,
                 field_distributions=None, default_distribution=None,
                 categorical_transformer=None, rounding='auto', min_value='auto',
                 max_value='auto'):
        """
        对联邦高斯分布进行初始化
        """

        # 初始化标记
        self._metadata_fitted = False

        # 初始化结构
        self._field_distributions = {}

        if table_metadata is None:
            self._metadata = Table(
                field_names=field_names,
                field_types=field_types,
                field_transformers=field_transformers,
                # dtype_transformers=self._DTYPE_TRANSFORMERS,
                # rounding=rounding,
                # min_value=min_value,
                # max_value=max_value
            )
            self._metadata_fitted = False
        else:
            table_metadata = deepcopy(table_metadata)
            for arg in (field_names, primary_key, field_types, anonymize_fields, constraints):
                if arg:
                    raise ValueError(
                        'If table_metadata is given {} must be None'.format(arg.__name__))

            if isinstance(table_metadata, dict):
                table_metadata = Table.from_dict(table_metadata)

            table_metadata.dtype_transformers.update(self._DTYPE_TRANSFORMERS)

            self._metadata = table_metadata
            self._metadata_fitted = table_metadata.fitted

        # 初始化默认类
        self._default_distribution = "default"
        self.privacy = None

    def fit(self, data_list, discrete_columns):
        """

        :param data_list:
        :return:
        """
        # data_list中每个元素是否都是DataFrame
        if not isinstance(data_list, list):
            raise ValueError('fed_gc need a data list.')
        client_num = len(data_list)
        assert client_num > 1, "data_list中至少要包含两个及以上的同构数据集"
        for i in range(client_num):
            if not isinstance(data_list[i], pd.DataFrame):
                raise ValueError('every clien\'s data should be Pandas DataFrame.')

        # example_data = data_list[0]
        example_data = pd.concat(data_list)

        # 获取metadata
        if not self._metadata_fitted:
            self._metadata.fit(example_data, discrete_columns)

        # 获取transformer
        LOGGER.info('Fitting HyperTransformer for table %s', self._metadata.name)
        self._metadata._fit_hyper_transformer(data_list)

        # 检查获取的metadata是否与其他的节点符合
        # for i in range(1, client_num):
        #     if not self._metadata.check(data_list[i]):
        #         raise ValueError('fed_gc need a data list.')

        # transformed_list = []
        # for i in range(client_num):
        #     transformed_list.append(self._metadata._hyper_transformer.transform(data_list[i]))
        # 后续要修改成真分布式的
        transformed_list = self._metadata._hyper_transformer.transform(data_list)
        # num_all = len(transformed)
        # transformed_list = [
        #     transformed[:int(num_all*0.15)],
        #     transformed[int(num_all * 0.15):int(num_all * 0.45)],
        #     transformed[int(num_all * 0.45):],
        # ]
        # transformed_list = devide_list(transformed, data_list)

        for column in example_data.columns:
            if column not in self._field_distributions:
                # Check if the column is a derived column.
                # column_name = column.replace('.value', '')
                distribution = 'gaussian' if self._metadata.get_dtypes()[column] == 'object' else 'gaussian mixed'

                self._field_distributions[column] = self._field_distributions.get(
                    column, distribution)

        self._model = Multivariate(distribution=self._field_distributions)
        LOGGER.debug('Fitting %s to table %s; shape: %s', self._model.__class__.__name__,
                     self._metadata.name, [table.shape for table in data_list])

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', module='scipy')
            self._model.fit(transformed_list)

    def validate_random_state(random_state):
        """Validate random state argument.

        Args:
            random_state (int, numpy.random.RandomState, tuple, or None):
                Seed or RandomState for the random generator.

        Output:
            numpy.random.RandomState
        """
        if random_state is None:
            return None

        if isinstance(random_state, int):
            return np.random.RandomState(seed=random_state)
        elif isinstance(random_state, np.random.RandomState):
            return random_state
        else:
            raise TypeError(
                f'`random_state` {random_state} expected to be an int '
                'or `np.random.RandomState` object.')

    def _randomize_samples(self, randomize_samples):
        """Randomize the samples according to user input.

        If ``randomize_samples`` is false, fix the seed that the random number generator
        uses in the underlying models.

        Args:
            randomize_samples (bool):
                Whether or not to randomize the generated samples.
        """
        if self._model is None:
            return

        if randomize_samples:
            # self._set_random_state(None)
            self.random_state = validate_random_state(None)
        else:
            # self._set_random_state(FIXED_RNG_SEED)
            self.random_state = validate_random_state(FIXED_RNG_SEED)

    def _validate_conditions(self, conditions):
        """Validate the user-passed conditions."""
        for column in conditions.columns:
            if column not in self._metadata.get_fields():
                raise ValueError(f'Unexpected column name `{column}`. '
                                 f'Use a column name that was present in the original data.')

    def sample(self, num_rows, randomize_samples=True, batch_size=None, output_file_path='.sample.csv.temp',
               conditions=None):
        if conditions is not None:
            raise TypeError('This method does not support the conditions parameter. '
                            'Please create `sdv.sampling.Condition` objects and pass them '
                            'into the `sample_conditions` method. '
                            'See User Guide or API for more details.')

        if num_rows is None:
            raise ValueError('You must specify the number of rows to sample (e.g. num_rows=100).')

        if num_rows == 0:
            return pd.DataFrame()

        ## 正式开始数据生成
        self._randomize_samples(randomize_samples)

        batch_size = min(batch_size, num_rows) if batch_size else num_rows

        sampled = []
        try:
            def _sample_function(progress_bar=None):
                for step in range(math.ceil(num_rows / batch_size)):
                    sampled_rows = self._sample_batch(
                        batch_size,
                        batch_size_per_try=batch_size,
                        progress_bar=progress_bar,
                        output_file_path=output_file_path,
                    )
                    sampled.append(sampled_rows)

                return sampled

            if batch_size == num_rows:
                sampled = _sample_function()
            else:
                sampled = progress_bar_wrapper(_sample_function, num_rows, 'Sampling rows')

        except (Exception, KeyboardInterrupt) as error:
            handle_sampling_error(output_file_path == TMP_FILE_NAME, output_file_path, error)

        else:
            if output_file_path == TMP_FILE_NAME and os.path.exists(output_file_path):
                os.remove(output_file_path)

        return pd.concat(sampled, ignore_index=True) if len(sampled) > 0 else pd.DataFrame()

    def set_privacy(self, epsilon, delta):
        self.privacy = {
            'epsilon': epsilon,
            'delta': delta
        }

    def sample_remaining_columns(self, known_columns, max_tries=100, batch_size_per_try=None, randomize_samples=True,
                                 output_file_path=None):
        """
        Sample rows from this table.
        """

        self._randomize_samples(randomize_samples)
        known_columns = known_columns.copy()
        self._validate_conditions(known_columns)

        sampled = pd.DataFrame()
        try:
            def _sample_function(progress_bar=None):
                return self._sample_with_conditions(
                    known_columns, max_tries, batch_size_per_try, progress_bar, output_file_path)

            if len(known_columns) == 1 and max_tries == 1:
                sampled = _sample_function()
            else:
                sampled = progress_bar_wrapper(
                    _sample_function, len(known_columns), 'Sampling remaining columns')
        except (Exception, KeyboardInterrupt) as error:
            handle_sampling_error(output_file_path == TMP_FILE_NAME, output_file_path, error)

        return sampled

    def _sample_with_conditions(self, conditions, max_tries, batch_size_per_try,
                                progress_bar=None, output_file_path=None):
        try:
            transformed_conditions = self._metadata.transform(conditions, on_missing_column='drop')
        except ConstraintsNotMetError as cnme:
            cnme.message = 'Provided conditions are not valid for the given constraints'
            raise

        condition_columns = list(conditions.columns)
        transformed_columns = list(transformed_conditions.columns)
        conditions.index.name = COND_IDX
        conditions.reset_index(inplace=True)
        transformed_conditions.index.name = COND_IDX
        # transformed_conditions.reset_index(inplace=True)
        grouped_conditions = conditions.groupby(condition_columns)

        # sample
        all_sampled_rows = list()

        for group, dataframe in grouped_conditions:
            if not isinstance(group, tuple):
                group = [group]

            condition_indices = dataframe[COND_IDX]
            condition = dict(zip(condition_columns, group))
            if len(transformed_columns) == 0:
                sampled_rows = self._conditionally_sample_rows(
                    dataframe,
                    condition,
                    None,
                    max_tries,
                    batch_size_per_try,
                    progress_bar=progress_bar,
                    output_file_path=output_file_path,
                )
                all_sampled_rows.append(sampled_rows)
            else:
                transformed_conditions_in_group = transformed_conditions.loc[condition_indices]
                transformed_groups = transformed_conditions_in_group.groupby(transformed_columns)
                for transformed_group, transformed_dataframe in transformed_groups:
                    if not isinstance(transformed_group, tuple):
                        transformed_group = [transformed_group]

                    transformed_condition = dict(zip(transformed_columns, transformed_group))
                    sampled_rows = self._conditionally_sample_rows(
                        transformed_dataframe,
                        condition,
                        transformed_condition,
                        max_tries,
                        batch_size_per_try,
                        progress_bar=progress_bar,
                        output_file_path=output_file_path,
                    )
                    all_sampled_rows.append(sampled_rows)

        all_sampled_rows = pd.concat(all_sampled_rows)
        if len(all_sampled_rows) == 0:
            return all_sampled_rows

        # all_sampled_rows = all_sampled_rows.set_index(COND_IDX)
        # all_sampled_rows.index.name = conditions.index.name
        # all_sampled_rows = all_sampled_rows.sort_index()
        # all_sampled_rows = self._metadata.make_ids_unique(all_sampled_rows)

        return all_sampled_rows

    def _conditionally_sample_rows(self, dataframe, condition, transformed_condition,
                                   max_tries=None, batch_size_per_try=None, float_rtol=0.01,
                                   graceful_reject_sampling=True, progress_bar=None,
                                   output_file_path=None):
        num_rows = len(dataframe)
        sampled_rows = self._sample_batch(
            num_rows,
            max_tries,
            batch_size_per_try,
            condition,
            transformed_condition,
            float_rtol,
            progress_bar,
            output_file_path,
        )

        # if len(sampled_rows) > 0:
        #     sampled_rows[COND_IDX] = dataframe[COND_IDX].values[:len(sampled_rows)]
        # else:
        #     # Didn't get any rows.
        #     if not graceful_reject_sampling:
        #         user_msg = ('Unable to sample any rows for the given conditions '
        #                     f'`{transformed_condition}`. ')
        #         if hasattr(self, '_model') and isinstance(
        #                 self._model, copulas.multivariate.GaussianMultivariate):
        #             user_msg = user_msg + (
        #                 'This may be because the provided values are out-of-bounds in the '
        #                 'current model. \nPlease try again with a different set of values.'
        #             )
        #         else:
        #             user_msg = user_msg + (
        #                 f'Try increasing `max_tries` (currently: {max_tries}) or increasing '
        #                 f'`batch_size_per_try` (currently: {batch_size_per_try}). Note that '
        #                 'increasing these values will also increase the sampling time.'
        #             )
        #
        #         raise ValueError(user_msg)

        return sampled_rows

    def _sample_batch(self, num_rows=None, max_tries=100, batch_size_per_try=None,
                      conditions=None, transformed_conditions=None, float_rtol=0.01,
                      progress_bar=None, output_file_path=None):

        if not batch_size_per_try:
            batch_size_per_try = num_rows * 10

        counter = 0
        num_valid = 0
        prev_num_valid = None
        remaining = num_rows
        sampled = pd.DataFrame()

        while num_valid < num_rows:
            if counter >= max_tries:
                break

            prev_num_valid = num_valid
            sampled, num_valid = self._sample_rows(
                batch_size_per_try, conditions, transformed_conditions, float_rtol, sampled,
            )

            # num_increase = min(num_valid - prev_num_valid, remaining)
            # if num_increase > 0:
            #     if output_file_path:
            #         append_kwargs = {'mode': 'a', 'header': False} if os.path.getsize(
            #             output_file_path) > 0 else {}
            #         sampled.head(min(len(sampled), num_rows)).tail(num_increase).to_csv(
            #             output_file_path,
            #             index=False,
            #             **append_kwargs,
            #         )
            #     if progress_bar:
            #         progress_bar.update(num_increase)

            # remaining = num_rows - num_valid
            # if remaining > 0:
            #     LOGGER.info(
            #         f'{remaining} valid rows remaining. Resampling {batch_size_per_try} rows')
            #
            # counter += 1

        return sampled.head(min(len(sampled), num_rows))

    def _sample_rows(self, num_rows, conditions=None, transformed_conditions=None,
                     float_rtol=0.1, previous_rows=None):

        if self._metadata.get_dtypes(ids=False):
            if conditions is None:
                sampled = self._sample(num_rows)
            else:
                try:
                    sampled = self._sample(num_rows, transformed_conditions)
                except NotImplementedError:
                    sampled = self._sample(num_rows)

            sampled = self._metadata.reverse_transform(sampled)

            if previous_rows is not None:
                sampled = pd.concat([previous_rows, sampled], ignore_index=True)

            # sampled = self._metadata.filter_valid(sampled)

            num_valid = len(sampled)

            return sampled, num_valid

        else:
            sampled = pd.DataFrame(index=range(num_rows))
            sampled = self._metadata.reverse_transform(sampled)
            return sampled, num_rows

    def _sample(self, num_rows=1, conditions=None):

        samples = self._get_normal_samples(num_rows, conditions)

        output = {}

        # 原始数据的column数量对应等量的univariate对象
        # 每个univariate对应不等量的samples中的列，但经过转换会变成一列
        for column_name, univariate in zip(self._model.columns, self._model.univariates):
            if conditions and column_name in conditions:
                # Use the values that were given as conditions in the original space.
                output[column_name] = np.full(num_rows, conditions[column_name])
            else:
                if univariate.type == "gmm":
                    normalized_value = samples[univariate.new_column_name[0]]
                    component_value = samples[univariate.new_column_name[1]]
                    # normalized_cdf = stats.norm.cdf(normalized_value)
                    # component_cdf = stats.norm.cdf(component_value)
                    output[column_name] = univariate.reverse_gmm_transform(
                        np.stack([
                            normalized_value,
                            component_value
                        ], axis=1),
                        column_name
                    )
                elif univariate.type == "gaussian":
                    cdf = stats.norm.cdf(samples[column_name])
                    output[column_name] = univariate.percent_point(cdf)

        return pd.DataFrame(data=output)

    def _get_normal_samples(self, num_rows, conditions):

        if conditions is None:
            covariance = self._model.covariance
            columns = self._model.new_columns
            means = np.zeros(len(columns))
        else:
            conditions = pd.Series(conditions)
            normal_conditions = self.norm_column(conditions)[0]
            normal_conditions = pd.Series(normal_conditions, index=conditions.index)
            means, covariance, columns = self._get_conditional_distribution(normal_conditions)

        if self.privacy is not None:
            delta = self.privacy['delta']
            epsilon = self.privacy['epsilon']
            sigma = 2 * np.power(2 * np.log(2/delta), 0.5) / epsilon

            for i in range(covariance.shape[0]):
                for j in range(covariance.shape[1]):
                    if i != j:
                        covariance.iloc[i][j] += np.random.normal(0, sigma)
                        covariance.iloc[j][i] += np.random.normal(0, sigma)
            print("sigma: ", sigma)

        samples = np.random.multivariate_normal(means, covariance, size=num_rows)
        return pd.DataFrame(samples, columns=columns)

    def norm_column(self, X):

        if isinstance(X, pd.Series):
            X = X.to_frame().T
        elif not isinstance(X, pd.DataFrame):
            if len(X.shape) == 1:
                X = [X]

        U = []
        for column_name, univariate in zip(self._model.columns, self._model.univariates):
            if column_name in X:
                column = X[column_name]
                U.append(univariate.norm(column, None))

        return np.column_stack(U)

    def _transform_to_normal(self, X):
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        elif not isinstance(X, pd.DataFrame):
            if len(X.shape) == 1:
                X = [X]

            X = pd.DataFrame(X, columns=self._model.columns)

        U = []
        for column_name, univariate in zip(self._model.columns, self._model.univariates):
            if column_name in X:
                column = X[column_name]
                U.append(univariate.cdf(column.to_numpy()).clip(EPSILON, 1 - EPSILON))

        return stats.norm.ppf(np.column_stack(U))

    def _get_conditional_distribution(self, conditions):
        columns2 = conditions.index
        columns1 = self._model.covariance.columns.difference(columns2)

        sigma11 = self._model.covariance.loc[columns1, columns1].to_numpy()
        sigma12 = self._model.covariance.loc[columns1, columns2].to_numpy()
        sigma21 = self._model.covariance.loc[columns2, columns1].to_numpy()
        sigma22 = self._model.covariance.loc[columns2, columns2].to_numpy()

        mu1 = np.zeros(len(columns1))
        mu2 = np.zeros(len(columns2))

        sigma12sigma22inv = sigma12 @ np.linalg.inv(sigma22)

        mu_bar = mu1 + sigma12sigma22inv @ (conditions - mu2)
        sigma_bar = sigma11 - sigma12sigma22inv @ sigma21

        return mu_bar, sigma_bar, columns1
