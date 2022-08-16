"""Base Univariate class."""

import pickle
from abc import ABC
from enum import Enum
import numpy as np
import pandas as pd
import rdt
import scipy
from scipy.optimize import fmin_slsqp
from scipy import stats
from scipy.stats import _distn_infrastructure
from rdt.transformers import BayesGMMTransformer
from fedtda.utils import (
    NotFittedError, get_instance, get_qualified_name,
    store_args, validate_random_state, valarray, argsreduce
)
from fedtda import Hyper_transformer
from fedtda import CategoricalTransformer
from fedtda import fed_compute
from copulas.univariate.selection import select_univariate

EPSILON = np.finfo(np.float32).eps
TRUNCNORM_TAIL_X = 30
_norm_pdf_C = np.sqrt(2*np.pi)
_norm_pdf_logC = np.log(_norm_pdf_C)


class ParametricType(Enum):
    """Parametric Enum."""

    NON_PARAMETRIC = 0
    PARAMETRIC = 1


class BoundedType(Enum):
    """Bounded Enum."""

    UNBOUNDED = 0
    SEMI_BOUNDED = 1
    BOUNDED = 2


class Univariate(object):
    """Univariate Distribution.

    Args:
        candidates (list[str or type or Univariate]):
            List of candidates to select the best univariate from.
            It can be a list of strings representing Univariate FQNs,
            or a list of Univariate subclasses or a list of instances.
        parametric (ParametricType):
            If not ``None``, only select subclasses of this type.
            Ignored if ``candidates`` is passed.
        bounded (BoundedType):
            If not ``None``, only select subclasses of this type.
            Ignored if ``candidates`` is passed.
        random_state (int or np.random.RandomState):
            Random seed or RandomState to use.
        selection_sample_size (int):
            Size of the subsample to use for candidate selection.
            If ``None``, all the data is used.
    """

    PARAMETRIC = ParametricType.NON_PARAMETRIC
    BOUNDED = BoundedType.UNBOUNDED

    fitted = False
    _constant_value = None
    _instance = None

    @classmethod
    def _select_candidates(cls, parametric=None, bounded=None):
        """Select which subclasses fulfill the specified constriants.

        Args:
            parametric (ParametricType):
                If not ``None``, only select subclasses of this type.
            bounded (BoundedType):
                If not ``None``, only select subclasses of this type.

        Returns:
            list:
                Selected subclasses.
        """
        candidates = []
        for subclass in cls.__subclasses__():
            candidates.extend(subclass._select_candidates(parametric, bounded))
            if ABC in subclass.__bases__:
                continue
            if parametric is not None and subclass.PARAMETRIC != parametric:
                continue
            if bounded is not None and subclass.BOUNDED != bounded:
                continue

            candidates.append(subclass)

        return candidates

    @store_args
    def __init__(self, candidates=None, parametric=None, bounded=None, random_state=None,
                 selection_sample_size=None, minimum=None, maximum=None):
        self.candidates = candidates or self._select_candidates(parametric, bounded)
        self.random_state = validate_random_state(random_state)
        self.selection_sample_size = selection_sample_size

        self.min = minimum
        self.max = maximum
        self.gmm = None
        self.transformed_data = None

        self._params = {
            'a': 0.0,
            'b': 0.0,
            'loc': 0.0,
            'scale': 0.0
        }

    @classmethod
    def __repr__(cls):
        """Return class name."""
        return cls.__name__

    def check_fit(self):
        """Check whether this model has already been fit to a random variable.

        Raise a ``NotFittedError`` if it has not.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        if not self.fitted:
            raise NotFittedError('This model is not fitted.')

    def _constant_sample(self, num_samples):
        """Sample values for a constant distribution.

        Args:
            num_samples (int):
                Number of rows to sample

        Returns:
            numpy.ndarray:
                Sampled values. Array of shape (num_samples,).
        """
        return np.full(num_samples, self._constant_value)

    def _constant_cumulative_distribution(self, X):
        """Cumulative distribution for the degenerate case of constant distribution.

        Note that the output of this method will be an array whose unique values are 0 and 1.
        More information can be found here: https://en.wikipedia.org/wiki/Degenerate_distribution

        Arguments:
            X (numpy.ndarray):
                Values for which the cumulative distribution will be computed.
                It must have shape (n, 1).

        Returns:
            numpy.ndarray:
                Cumulative distribution values for points in X.
        """
        result = np.ones(X.shape)
        result[np.nonzero(X < self._constant_value)] = 0

        return result

    def _constant_probability_density(self, X):
        """Probability density for the degenerate case of constant distribution.

        Note that the output of this method will be an array whose unique values are 0 and 1.
        More information can be found here: https://en.wikipedia.org/wiki/Degenerate_distribution

        Arguments:
            X (numpy.ndarray):
                Values for which the probability density will be computed.
                It must have shape (n, 1).

        Returns:
            numpy.ndarray:
                Probability density values for points in X.
        """
        result = np.zeros(X.shape)
        result[np.nonzero(X == self._constant_value)] = 1

        return result

    def _constant_percent_point(self, X):
        """Percent point for the degenerate case of constant distribution.

        Note that the output of this method will be an array whose unique values are `np.nan`
        and self._constant_value.
        More information can be found here: https://en.wikipedia.org/wiki/Degenerate_distribution

        Arguments:
            U (numpy.ndarray):
                Values for which the cumulative distribution will be computed.
                It must have shape (n, 1) and values must be in [0,1].

        Returns:
            numpy.ndarray:
                Inverse cumulative distribution values for points in U.
        """
        return np.full(X.shape, self._constant_value)

    def _replace_constant_methods(self):
        """Replace conventional distribution methods by its constant counterparts."""
        self.cumulative_distribution = self._constant_cumulative_distribution
        self.percent_point = self._constant_percent_point
        self.probability_density = self._constant_probability_density
        self.sample = self._constant_sample

    def _set_constant_value(self, constant_value):
        """Set the distribution up to behave as a degenerate distribution.

        The constant value is stored as ``self._constant_value`` and all
        the methods are replaced by their degenerate counterparts.

        Args:
            constant_value (float):
                Value to set as the constant one.
        """
        self._constant_value = constant_value
        self._replace_constant_methods()

    def _check_constant_value(self, X):
        """Check if a Series or array contains only one unique value.

        If it contains only one value, set the instance up to behave accordingly.

        Args:
            X (numpy.ndarray):
                Data to analyze.

        Returns:
            float:
                Whether the input data had only one value or not.
        """
        uniques = np.unique(X)
        if len(uniques) == 1:
            self._set_constant_value(uniques[0])

            return True

        return False

    def fed_fit(self, x_list, column_name, distribution):
        """
        """
        if distribution == 'default' or distribution == 'gaussian mixed':
            self.fit_gmm(x_list, column_name)
        elif distribution == 'gaussian':

            columns_list = []
            clients_num = len(x_list)
            for i in range(clients_num):
                column = x_list[i][column_name]
                columns_list.append(column)

            self.fit_gaussian(columns_list)

        else:
            self.fit_tg(x_list)

        self.fitted = True

    def fit_gaussian(self, column_list):
        """
        对分布数据计算高斯分布
        :param column_list:
        :return:
        """

        self._params = {
            'loc': fed_compute.fed_mean(column_list),
            'scale': fed_compute.fed_std(column_list)
        }

        self.type = "gaussian"

        self.transformed_data = pd.concat(column_list)

    def fit_gmm(self, x_list, column_name):
        """
        默认使用高斯混合分布
        :param x_list:
        :return:
        """
        gm = BayesGMMTransformer(max_clusters=1)
        # gm = BayesGMMTransformer()
        gm.fit(pd.concat(x_list), column_name)
        num_components = sum(gm.valid_component_indicator)
        self.gmm = gm

        # 目前已经有fed_gmm算法，因此这里我们直接用集中场景来模拟
        transformed = gm.transform(pd.concat(x_list), column_name)
        # transform以后得数据处理列变成了新的两列数据
        # "{}.normalized"和"{}.component"

        transformers_dict = {
            column_name+".component": CategoricalTransformer.CategoricalTransformer(fuzzy=True)
        }
        self.gmm_hyper_transformer = Hyper_transformer.HyperTransformer(field_transformers=transformers_dict)
        self.gmm_hyper_transformer.fit([transformed[[column_name+".component"]]])
        transformed_component = self.gmm_hyper_transformer.transform([transformed[[column_name+".component"]]])[0]
        # transform以后对应列名字增加".value"

        self.new_column_name = [column_name+".normalized", column_name+".component.value"]

        self.transformed_data = pd.concat([
            transformed[[column_name+".normalized"]],
            transformed_component
        ], axis=1)

        self.type = "gmm"

        self._params = {
            'normalized_loc': np.mean(self.transformed_data[[column_name + ".normalized"]]),
            'normalized_scale': np.std(self.transformed_data[[column_name + ".normalized"]]),
            'component_loc': np.mean(self.transformed_data[[column_name + ".component.value"]]),
            'component_scale': np.std(self.transformed_data[[column_name + ".component.value"]])
        }

    def reverse_gmm_transform(self, transformed_data, column_name):
        if not isinstance(transformed_data, pd.DataFrame):
            transformed_data = pd.DataFrame({
                self.new_column_name[0]: transformed_data[:, 0],
                self.new_column_name[1]: transformed_data[:, 1]
            })

        # component_column_name = self.new_column_name[1]
        transformed_data = self.gmm_hyper_transformer.reverse_transform(transformed_data)

        inversed_data = np.zeros(len(transformed_data))
        for i in range(len(transformed_data)):
            for n in range(self.gmm._bgm_transformer.n_components):
                if transformed_data.iloc[i,1] == n:
                    mean = self.gmm._bgm_transformer.means_[n]
                    scale_2 = self.gmm._bgm_transformer.covariances_[n].squeeze()
                    inversed_data[i] = transformed_data.iloc[i, 0] * np.sqrt(scale_2) + mean

        return inversed_data
        # return self.gmm.reverse_transform(transformed_data, [
        #     column_name + '.normalized',
        #     column_name + '.component'
        # ]).values.squeeze()

    def fit_tg(self, x_list):
        """
        使用了截断高斯分布（后续可能适配多种分布）
        :param x_list:
        :return:
        """
        if self.min is None:
            # self.min = X.min() - EPSILON
            clients_min_list = []
            for x in x_list:
                clients_min_list.append(x.min())
            self.min = fed_compute.fed_min(clients_min_list) - EPSILON

        if self.max is None:
            # self.max = X.max() + EPSILON
            clients_max_list = []
            for x in x_list:
                clients_max_list.append(x.max())
            self.max = fed_compute.fed_max(clients_max_list) - EPSILON

        def nnlf(params, local_x):
            """
            负对数似然函数
            """
            loc_i, scale_i = params
            a_i = (self.min - loc_i) / scale_i
            b_i = (self.max - loc_i) / scale_i

            norm_x = np.asarray((local_x - loc_i) / scale_i)
            n_log_scale = len(norm_x) * np.log(scale_i)

            return -np.sum(truncated_gaussian_log_pdf(norm_x, a_i, b_i), axis=0) + n_log_scale

        def fed_nnlf(params):
            nnlf_list = [nnlf(params, x) for x in x_list]
            return np.sum(nnlf_list)

        initial_params = fed_compute.fed_mean(x_list), fed_compute.fed_std(x_list)
        optimal = fmin_slsqp(fed_nnlf, initial_params, iprint=False, bounds=[
            (self.min, self.max),
            (0.0, (self.max - self.min)**2)
        ])

        loc, scale = optimal
        a = (self.min - loc) / scale
        b = (self.max - loc) / scale

        self._params = {
            'a': a,
            'b': b,
            'loc': loc,
            'scale': scale
        }

    def norm(self, x, start):
        if len(self._params) == 2:
            norm = (x-self._params['loc'])/self._params['scale']
            # return stats.norm.cdf(norm)
            return stats.norm.ppf(stats.norm.cdf(norm))
        elif len(self._params) == 4:

            normalized_norm = self.transformed_data.iloc[:,0]
            if self._params['normalized_scale'].values != 0:
                normalized_norm = (normalized_norm-self._params['normalized_loc'].values)/self._params['normalized_scale'].values
                normalized_norm = stats.norm.ppf(stats.norm.cdf(normalized_norm))
            else:
                normalized_norm = (normalized_norm - self._params['normalized_loc'].values)
                # normalized_norm = stats.norm.ppf(stats.norm.cdf(normalized_norm))

            component_norm = self.transformed_data.iloc[:, 1]
            if self._params['component_scale'].values != 0:
                component_norm = (component_norm - self._params['component_loc'].values) / self._params['component_scale'].values
                component_norm = stats.norm.ppf(stats.norm.cdf(component_norm))
            else:
                component_norm = (component_norm - self._params['component_loc'].values)


            data_num = len(x)
            return np.stack([
                normalized_norm[start:start+data_num],
                component_norm[start:start+data_num]
            ], axis=1)

    def probability_density(self, X):
        """Compute the probability density for each point in X.

        Arguments:
            X (numpy.ndarray):
                Values for which the probability density will be computed.
                It must have shape (n, 1).

        Returns:
            numpy.ndarray:
                Probability density values for points in X.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        return self._instance.probability_density(X)

    def log_probability_density(self, X):
        """Compute the log of the probability density for each point in X.

        It should be overridden with numerically stable variants whenever possible.

        Arguments:
            X (numpy.ndarray):
                Values for which the log probability density will be computed.
                It must have shape (n, 1).

        Returns:
            numpy.ndarray:
                Log probability density values for points in X.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        if self._instance:
            return self._instance.log_probability_density(X)

        return np.log(self.probability_density(X))

    def pdf(self, X):
        """Compute the probability density for each point in X.

        Arguments:
            X (numpy.ndarray):
                Values for which the probability density will be computed.
                It must have shape (n, 1).

        Returns:
            numpy.ndarray:
                Probability density values for points in X.
        """
        return self.probability_density(X)

    def cumulative_distribution(self, X):
        """Compute the cumulative distribution value for each point in X.

        Arguments:
            X (numpy.ndarray):
                Values for which the cumulative distribution will be computed.
                It must have shape (n, 1).

        Returns:
            numpy.ndarray:
                Cumulative distribution values for points in X.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        transformed = self.norm(X)
        return stats.multivariate_normal.cdf(transformed, cov=self.covariance)

    def cdf(self, X):
        """Compute the cumulative distribution value for each point in X.

        Arguments:
            X (numpy.ndarray):
                Values for which the cumulative distribution will be computed.
                It must have shape (n, 1).

        Returns:
            numpy.ndarray:
                Cumulative distribution values for points in X.
        """
        return self.cumulative_distribution(X)

    def percent_point(self, U):
        """Compute the inverse cumulative distribution value for each point in U.

        Arguments:
            U (numpy.ndarray):
                Values for which the cumulative distribution will be computed.
                It must have shape (n, 1) and values must be in [0,1].

        Returns:
            numpy.ndarray:
                Inverse cumulative distribution values for points in U.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        ppf = stats.norm.ppf(U)
        return ppf * self._params['scale'] + self._params['loc']
        # return stats.truncnorm.ppf(U, **self._params)
        # return self.MODEL_CLASS.ppf(U, **self._params)

    def ppf(self, U):
        """Compute the inverse cumulative distribution value for each point in U.

        Arguments:
            U (numpy.ndarray):
                Values for which the cumulative distribution will be computed.
                It must have shape (n, 1) and values must be in [0,1].

        Returns:
            numpy.ndarray:
                Inverse cumulative distribution values for points in U.
        """
        return _distn_infrastructure._doc_ppf

        args = self._params["a"], self._params["b"]
        loc = self._params["loc"]
        scale = self._params["scale"]
        q, loc, scale = map(np.asarray, (U, loc, scale))
        args = tuple(map(np.asarray, args))
        _a, _b = args[0], args[1]
        cond0 = (scale > 0) & (loc == loc)
        cond1 = (0 < q) & (q < 1)
        cond2 = cond0 & (q == 0)
        cond3 = cond0 & (q == 1)
        cond = cond0 & cond1
        output = valarray(np.shape(cond), value=np.nan)

        lower_bound = _a * scale + loc
        upper_bound = _b * scale + loc
        np.place(output, cond2, argsreduce(cond2, lower_bound)[0])
        np.place(output, cond3, argsreduce(cond3, upper_bound)[0])

        if np.any(cond):  # call only if at least 1 entry
            goodargs = argsreduce(cond, *((q,) + args + (scale, loc)))
            scale, loc, goodargs = goodargs[-2], goodargs[-1], goodargs[:-2]
            np.place(output, cond, self._ppf(*goodargs) * scale + loc)
        if output.ndim == 0:
            return output[()]
        return output

    def _ppf(self, q, a, b):
        if np.isscalar(a) and np.isscalar(b):
            return _truncnorm_ppf_scalar(q, a, b)
        a, b = np.atleast_1d(a), np.atleast_1d(b)
        if a.size == 1 and b.size == 1:
            return _truncnorm_ppf_scalar(q, a.item(), b.item())

        out = None
        it = np.nditer([q, a, b, out], [],
                       [['readonly'], ['readonly'], ['readonly'], ['writeonly', 'allocate']])
        for (_q, _a, _b, _x) in it:
            _x[...] = _truncnorm_ppf_scalar(_q, _a, _b)
        return it.operands[3]

    def set_random_state(self, random_state):
        """Set the random state.

        Args:
            random_state (int, np.random.RandomState, or None):
                Seed or RandomState for the random generator.
        """
        self.random_state = validate_random_state(random_state)

    def sample(self, n_samples=1):
        """Sample values from this model.

        Argument:
            n_samples (int):
                Number of values to sample

        Returns:
            numpy.ndarray:
                Array of shape (n_samples, 1) with values randomly
                sampled from this model distribution.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        return self._instance.sample(n_samples)

    def _get_params(self):
        """Return attributes from self.model to serialize.

        Returns:
            dict:
                Parameters of the underlying distribution.
        """
        return self._instance._get_params()

    def _set_params(self, params):
        """Set the parameters of this univariate.

        Must be implemented in all the subclasses.

        Args:
            dict:
                Parameters to recreate this instance.
        """
        raise NotImplementedError()

    def to_dict(self):
        """Return the parameters of this model in a dict.

        Returns:
            dict:
                Dictionary containing the distribution type and all
                the parameters that define the distribution.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()

        params = self._get_params()
        if self.__class__ is Univariate:
            params['type'] = get_qualified_name(self._instance)
        else:
            params['type'] = get_qualified_name(self)

        return params

    @classmethod
    def from_dict(cls, params):
        """Build a distribution from its params dict.

        Args:
            params (dict):
                Dictionary containing the FQN of the distribution and the
                necessary parameters to rebuild it.
                The input format is exactly the same that is outputted by
                the distribution class ``to_dict`` method.

        Returns:
            Univariate:
                Distribution instance.
        """
        params = params.copy()
        distribution = get_instance(params.pop('type'))
        distribution._set_params(params)
        distribution.fitted = True

        return distribution

    def save(self, path):
        """Serialize this univariate instance using pickle.

        Args:
            path (str):
                Path to where this distribution will be serialized.
        """
        with open(path, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    @classmethod
    def load(cls, path):
        """Load a Univariate instance from a pickle file.

        Args:
            path (str):
                Path to the pickle file where the distribution has been serialized.

        Returns:
            Univariate:
                Loaded instance.
        """
        with open(path, 'rb') as pickle_file:
            return pickle.load(pickle_file)


def norm_logcdf(x):
    # return np.log(scipy.stats.norm.cdf(x))
    return -x ** 2 / 2.0 - _norm_pdf_logC


def truncated_gaussian_log_pdf(x, a, b):
    """

    :return:
    """
    if (a <= TRUNCNORM_TAIL_X) and (b >= -TRUNCNORM_TAIL_X):
        if a > 0:
            delta = scipy.stats.norm.cdf(-a) - scipy.stats.norm.cdf(-b)
        else:
            delta = scipy.stats.norm.cdf(b) - scipy.stats.norm.cdf(a)
        if delta > 0:
            return norm_logcdf(x) - np.log(delta)

    if b < 0 or (np.abs(a) >= np.abs(b)):
        nla, nlb = norm_logcdf(a), norm_logcdf(b)
        logdelta = nlb + np.log1p(-np.exp(nla - nlb))
    else:
        sla, slb = norm_logcdf(-a), norm_logcdf(-b)
        logdelta = sla + np.log1p(-np.exp(slb - sla))

    return norm_logcdf(x) - logdelta


def _truncnorm_ppf_scalar(q, a, b):
    shp = np.shape(q)
    q = np.atleast_1d(q)
    out = np.zeros(np.shape(q))
    condle0, condge1 = (q <= 0), (q >= 1)
    if np.any(condle0):
        out[condle0] = a
    if np.any(condge1):
        out[condge1] = b
    delta = _truncnorm_get_delta_scalar(a, b)
    cond_inner = ~condle0 & ~condge1
    if np.any(cond_inner):
        qinner = q[cond_inner]
        if delta > 0:
            if a > 0:
                sa, sb = _norm_sf(a), _norm_sf(b)
                np.place(out, cond_inner,
                         _norm_isf(qinner * sb + sa * (1.0 - qinner)))
            else:
                na, nb = _norm_cdf(a), _norm_cdf(b)
                np.place(out, cond_inner, _norm_ppf(qinner * nb + na * (1.0 - qinner)))
        elif np.isinf(b):
            np.place(out, cond_inner,
                     -_norm_ilogcdf(np.log1p(-qinner) + _norm_logsf(a)))
        elif np.isinf(a):
            np.place(out, cond_inner,
                     _norm_ilogcdf(np.log(q) + _norm_logcdf(b)))
        else:
            if b < 0:
                # Solve norm_logcdf(x) = norm_logcdf(a) + log1p(q * (expm1(norm_logcdf(b)  - norm_logcdf(a)))
                #      = nla + log1p(q * expm1(nlb - nla))
                #      = nlb + log(q) + log1p((1-q) * exp(nla - nlb)/q)
                def _f_cdf(x, c):
                    return _norm_logcdf(x) - c

                nla, nlb = _norm_logcdf(a), _norm_logcdf(b)
                values = nlb + np.log(q[cond_inner])
                C = np.exp(nla - nlb)
                if C:
                    one_minus_q = (1 - q)[cond_inner]
                    values += np.log1p(one_minus_q * C / q[cond_inner])
                x = [optimize.zeros.brentq(_f_cdf, a, b, args=(c,),
                                           maxiter=TRUNCNORM_MAX_BRENT_ITERS)for c in values]
                np.place(out, cond_inner, x)
            else:
                # Solve norm_logsf(x) = norm_logsf(b) + log1p((1-q) * (expm1(norm_logsf(a)  - norm_logsf(b)))
                #      = slb + log1p((1-q)[cond_inner] * expm1(sla - slb))
                #      = sla + log(1-q) + log1p(q * np.exp(slb - sla)/(1-q))
                def _f_sf(x, c):
                    return _norm_logsf(x) - c

                sla, slb = _norm_logsf(a), _norm_logsf(b)
                one_minus_q = (1-q)[cond_inner]
                values = sla + np.log(one_minus_q)
                C = np.exp(slb - sla)
                if C:
                    values += np.log1p(q[cond_inner] * C / one_minus_q)
                x = [optimize.zeros.brentq(_f_sf, a, b, args=(c,),
                                             maxiter=TRUNCNORM_MAX_BRENT_ITERS) for c in values]
                np.place(out, cond_inner, x)
        out[out < a] = a
        out[out > b] = b
    return (out[0] if (shp == ()) else out)


def _truncnorm_get_delta_scalar(a, b):
    if (a > TRUNCNORM_TAIL_X) or (b < -TRUNCNORM_TAIL_X):
        return 0
    if a > 0:
        delta = _norm_sf(a) - _norm_sf(b)
    else:
        delta = _norm_cdf(b) - _norm_cdf(a)
    delta = max(delta, 0)
    return delta
