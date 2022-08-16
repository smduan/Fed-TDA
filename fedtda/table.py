import rdt
import logging
import numpy as np
import copy
import operator
import pandas as pd
from fedtda.Hyper_transformer import HyperTransformer
from fedtda.CategoricalTransformer import CategoricalTransformer
from fedtda.numerical import NumericalTransformer
LOGGER = logging.getLogger(__name__)


class Table:
    # 初始化时默认的属性
    _fields_metadata = None
    fitted = False

    _DTYPE_TRANSFORMERS = {
        'i': 'integer',
        'f': 'float',
        'O': 'categorical_fuzzy',
        'b': 'boolean',
        'M': 'datetime',
    }

    _DTYPES_TO_TYPES = {
        'i': {
            'type': 'numerical',
            'subtype': 'integer',
        },
        'f': {
            'type': 'numerical',
            'subtype': 'float',
        },
        'O': {
            'type': 'categorical',
        },
        'b': {
            'type': 'boolean',
        },
        'M': {
            'type': 'datetime',
        }
    }
    _TYPES_TO_DTYPES = {
        ('categorical', None): 'object',
        ('boolean', None): 'bool',
        ('numerical', None): 'float',
        ('numerical', 'float'): 'float',
        ('numerical', 'integer'): 'int',
        ('datetime', None): 'datetime64',
        ('id', None): 'int',
        ('id', 'integer'): 'int',
        ('id', 'string'): 'str'
    }

    _TRANSFORMER_TEMPLATES = {
        # 'integer': rdt.transformers.NumericalTransformer(dtype=int, min_value='auto', max_value='auto'),
        'integer': NumericalTransformer(dtype=int, min_value='auto', max_value='auto'),
        # 'float': rdt.transformers.NumericalTransformer(dtype=float, min_value='auto', max_value='auto'),
        'float': NumericalTransformer(dtype=float, min_value='auto', max_value='auto'),
        # 'categorical': rdt.transformers.CategoricalTransformer,
        # 'categorical_fuzzy': rdt.transformers.CategoricalTransformer(fuzzy=True),
        'categorical_fuzzy': CategoricalTransformer(fuzzy=True),
        'one_hot_encoding': rdt.transformers.OneHotEncodingTransformer,
        'label_encoding': rdt.transformers.LabelEncodingTransformer,
        'boolean': rdt.transformers.BooleanTransformer,
        'datetime': rdt.transformers.DatetimeTransformer(strip_constant=True),
    }

    def __init__(self, name=None, field_names=None, field_types=None, field_transformers=None,):
        """

        """
        self.name = name
        self._field_names = field_names
        self._dtypes = None
        self._field_types = field_types or {}
        self._field_transformers = field_transformers or {}

        self.dtype_transformers = self._DTYPE_TRANSFORMERS.copy()
        self._transformer_templates = self._TRANSFORMER_TEMPLATES.copy()

    def get_dtypes(self, ids=False):

        dtypes = dict()
        for name, field_meta in self._fields_metadata.items():
            field_type = field_meta['type']

            if ids or (field_type != 'id'):
                dtypes[name] = self._get_field_dtype(name, field_meta)

        return dtypes

    def transform(self, data, on_missing_column='error'):
        LOGGER.debug('Transforming table %s', self.name)
        try:
            return self._hyper_transformer.transform(data)
        except rdt.errors.NotFittedError:
            return data

    def get_fields(self):
        """Get fields metadata.

        Returns:
            dict:
                Dictionary of fields metadata for this table.
        """
        return copy.deepcopy(self._fields_metadata)

    def _get_field_dtype(self, field_name, field_metadata):
        field_type = field_metadata['type']
        field_subtype = field_metadata.get('subtype')
        dtype = self._TYPES_TO_DTYPES.get((field_type, field_subtype))
        if not dtype:
            raise MetadataError(
                'Invalid type and subtype combination for field {}: ({}, {})'.format(
                    field_name, field_type, field_subtype)
            )

        return dtype

    def _build_fields_metadata(self, data):
        fields_metadata = dict()
        for field_name in self._field_names:
            if field_name not in data:
                raise ValueError('Field {} not found in given data'.format(field_name))

            field_meta = self._field_types.get(field_name)
            if field_meta:
                dtype = self._get_field_dtype(field_name, field_meta)
            else:
                dtype = data[field_name].dtype
                field_template = self._DTYPES_TO_TYPES.get(dtype.kind)
                if field_template is None:
                    msg = 'Unsupported dtype {} in column {}'.format(dtype, field_name)
                    raise ValueError(msg)

                field_meta = copy.deepcopy(field_template)

            field_transformer = self._field_transformers.get(field_name)
            if field_transformer:
                field_meta['transformer'] = field_transformer
            else:
                field_meta['transformer'] = self.dtype_transformers.get(np.dtype(dtype).kind)

            fields_metadata[field_name] = field_meta

        return fields_metadata

    def _get_transformers(self, dtypes):
        transformers = dict()
        for name, dtype in dtypes.items():
            field_metadata = self._fields_metadata.get(name, {})
            transformer_template = field_metadata.get('transformer')
            if transformer_template is None:
                transformer_template = self.dtype_transformers[np.dtype(dtype).kind]
                if transformer_template is None:
                    # Skip this dtype
                    continue

                field_metadata['transformer'] = transformer_template

            if isinstance(transformer_template, str):
                transformer_template = self._transformer_templates[transformer_template]

            if isinstance(transformer_template, type):
                transformer = transformer_template()
            else:
                transformer = copy.deepcopy(transformer_template)

            LOGGER.debug('Loading transformer %s for field %s',
                         transformer.__class__.__name__, name)
            transformers[name] = transformer

        return transformers

    def _fit_hyper_transformer(self, data_list):
        meta_dtypes = self.get_dtypes(ids=False)

        dtypes = {}
        numerical_extras = []
        data_list = data_list.copy()
        data_example = pd.concat(data_list)

        for column in data_example.columns:
            if column in meta_dtypes:
                dtypes[column] = meta_dtypes[column]

        transformers_dict = self._get_transformers(dtypes)
        for column in numerical_extras:
            transformers_dict[column] = rdt.transformers.NumericalTransformer(min_value='auto', max_value='auto')

        self._hyper_transformer = HyperTransformer(field_transformers=transformers_dict)
        # self._hyper_transformer.fit(data_example[list(transformers_dict.keys())])
        self._hyper_transformer.fit(data_list)

    def reverse_transform(self, data):
        """Reverse the transformed data to the original format.

        Args:
            data (pandas.DataFrame):
                Data to be reverse transformed.

        Returns:
            pandas.DataFrame
        """
        # if not self.fitted:
        #     raise MetadataNotFittedError()

        try:
            reversed_data = self._hyper_transformer.reverse_transform(data)
        except rdt.errors.NotFittedError:
            reversed_data = data

        # for constraint in reversed(self._constraints):
        #     reversed_data = constraint.reverse_transform(reversed_data)

        for name, field_metadata in self._fields_metadata.items():
            field_type = field_metadata['type']
            if field_type == 'id' and name not in reversed_data:
                field_data = self._make_ids(field_metadata, len(reversed_data))
            elif field_metadata.get('pii', False):
                field_data = pd.Series(Table._get_fake_values(field_metadata, len(reversed_data)))
            else:
                field_data = reversed_data[name]

            reversed_data[name] = field_data[field_data.notnull()].astype(self._dtypes[name])

        return reversed_data[self._field_names]

    def fit(self, data, discrete_columns):
        """

        :param data:
        :return:
        """
        # 获取列名，分成两种情况
        # 如果filed_name是空，则用全部的columns
        # 如果filed_name不为空，则用filed_name与columns的交集
        LOGGER.info('Fitting table %s metadata', self.name)
        if not self._field_names:
            self._field_names = list(data.columns)
        elif isinstance(self._field_names, set):
            self._field_names = [field for field in data.columns if field in self._field_names]

        # 获取每一列的数据类型
        for column in data.columns:
            if column in discrete_columns:
                # data[column] = data[column].apply(str)
                data[column] = data[column].astype('object')
        self._dtypes = data[self._field_names].dtypes

        # 获取filed_matadet
        if not self._fields_metadata:
            self._fields_metadata = self._build_fields_metadata(data)

        self.fitted = True

    def check(self, data):
        """

        :param data:
        :return:
        """
        # 检查列名是否一致可以转set，然后比较字符串是否一致
        local_field_names = [field for field in data.columns if field in self._field_names]
        name_consistence = operator.__eq__(local_field_names, self._field_names)

        # 检查对英烈dtype是否一致
        local_dtypes = data[self._field_names].dtypes
        dtype_consistence = operator.__eq__(list(local_dtypes), list(self._dtypes))

        # 检查metadata是否一致
        local_fields_metadata = self._build_fields_metadata(data)
        metadata_consistence = operator.__eq__(local_fields_metadata, self._fields_metadata)

        return name_consistence and dtype_consistence and metadata_consistence
