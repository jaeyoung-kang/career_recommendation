import bisect
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder


class FeatureLabelEncoder:
    def fit_transform(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return self.transform(*args, **kwargs)
    
    def fit(self, *args, **kwargs):
        pass

    def transform(self, *args, **kwargs):
        pass

    def _add_unknown_value(self, encoder):
        data_type = encoder.classes_.dtype
        if data_type == 'O':
            unknown_value = '<unknown>'
        else:
            unknown_value = -9999

        encoder_classes = encoder.classes_.tolist()
        bisect.insort_left(encoder_classes, unknown_value)
        encoder.classes_ = np.array(encoder_classes)
        return encoder, unknown_value
    
    def _change_unknown_value(self, series, encoder, unknown):
        return series.map(lambda s: unknown if s not in encoder.classes_ else s)


class MultiFeatureLabelEncoder(FeatureLabelEncoder):
    def __init__(self):
        self._encoders = None
        self._unknowns = None
        
    def fit(self, data, features):
        self._encoders = {}
        self._unknowns = {}
        for feature in features:
            encoder = LabelEncoder()
            encoder.fit(data[feature])

            encoder, unknown = self._add_unknown_value(encoder)
            self._encoders[feature] = encoder
            self._unknowns[feature] = unknown

    def transform(self, data):
        data = data.copy()
        for feature, encoder in self._encoders.items():
            unknown = self._unknowns[feature]
            data[feature] = self._change_unknown_value(
                series=data[feature],
                encoder=encoder,
                unknown=unknown,
            )
            data[feature] = encoder.transform(data[feature]) + 1
        return data
    
    def inverse_transform(self, data):
        data = data.copy()
        for feature, encoder in self._encoders.items():
            data[feature] = encoder.inverse_transform(data[feature] - 1)
        return data


class VariableLenghthLabelEncoder(FeatureLabelEncoder):
    def __init__(self):
        self._encoder = None
        self._unknown = None
    
    def fit(self, series):
        series = series.explode()

        encoder = LabelEncoder()
        encoder.fit(series)

        encoder, unknown = self._add_unknown_value(encoder)
        self._encoder = encoder
        self._unknown = unknown
    
    def transform(self, series):
        series = series.copy()
        series = series.explode()
        index = series.index
        name = series.name

        series = self._change_unknown_value(
            series=series,
            encoder=self._encoder,
            unknown=self._unknown,
        )
        transformed_series = self._encoder.transform(series) + 1
        transformed_series = pd.Series(
            data=transformed_series,
            index=index,
            name=name,
        )
        
        reshaped_series = transformed_series.to_frame()
        reshaped_series = reshaped_series.reset_index().groupby('index')[name].unique()
        return reshaped_series
    
    def inverse_transform(self, series):
        series = series.copy()
        series = series.explode()
        index = series.index
        name = series.name

        series = pd.to_numeric(series)
        transformed_series = self._encoder.inverse_transform(series - 1)
        transformed_series = pd.Series(
            data=transformed_series,
            index=index,
            name=name,
        )

        reshaped_series = transformed_series.to_frame()
        reshaped_series = reshaped_series.reset_index().groupby('index')[name].unique()
        return reshaped_series
