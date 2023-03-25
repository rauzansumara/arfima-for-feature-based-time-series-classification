# Import Package R to Python
from rpy2.robjects import packages, vectors
forecast = packages.importr('forecast')

# Import Package Python
import numpy as np
import itertools
import pandas as pd

from sktime.datatypes._panel._convert import from_nested_to_2d_array, _concat_nested_arrays
from sktime.utils.validation.panel import check_X
import tsfresh.feature_extraction.feature_calculators as fc


# class BaseSeriesAsFeaturesTransformer
class BaseSeriesAsFeaturesTransformer():
    """
    Base class for transformers, for identification.
    """

    def fit(self, X, y=None):
        """
        empty fit function, which inheriting transformers can override
        if need be.
        """
        self._is_fitted = True
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it.
        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.
        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Training set.
        y : numpy array of shape [n_samples]
            Target values.
        Returns
        -------
        X_new : numpy array of shape [n_samples, n_features_new]
            Transformed array.
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        # fit method of arity 2 (supervised transformation)
        return self.fit(X, y, **fit_params).transform(X)


# class NotFittedError
class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting.
    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.

    References
    ----------
    ..[1] Based on scikit-learn's NotFittedError
    """


# class BaseEstimator
class BaseEstimator():

    # def __init__(self, *args, **kwargs):
    #     # Including args and kwargs make the class cooperative, so that args
    #     # and kwargs are passed on to other parent classes when using
    #     # multiple inheritance
    #     self._is_fitted = False
    #     super(BaseEstimator, self).__init__(*args, **kwargs)

    def __init__(self):
        self._is_fitted = False

    @property
    def is_fitted(self):
        """Has `fit` been called?"""
        return self._is_fitted

    def check_is_fitted(self):
        """Check if the estimator has been fitted.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        """
        if not self.is_fitted:
            raise NotFittedError(
                f"This instance of {self.__class__.__name__} has not "
                f"been fitted yet; please call `fit` first.")


# class BaseSeriesAsFeaturesTransformer
class BaseSeriesAsFeaturesTransformer(BaseEstimator):
    """
    Base class for transformers, for identification.
    """

    def fit(self, X, y=None):
        """
        empty fit function, which inheriting transformers can override
        if need be.
        """
        self._is_fitted = True
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it.
        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.
        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Training set.
        y : numpy array of shape [n_samples]
            Target values.
        Returns
        -------
        X_new : numpy array of shape [n_samples, n_features_new]
            Transformed array.
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        # fit method of arity 2 (supervised transformation)
        return self.fit(X, y, **fit_params).transform(X)


# class NonFittableSeriesAsFeaturesTransformer
class _NonFittableSeriesAsFeaturesTransformer(BaseSeriesAsFeaturesTransformer):
    """Base class for transformers which do nothing in fit and if fittable,
    fit during transform, otherwise only transform data"""
    pass


# class MetaEstimatorMixin
class MetaEstimatorMixin:
    _required_parameters = []


# class RowTransformer
class RowTransformer(_NonFittableSeriesAsFeaturesTransformer,
                     MetaEstimatorMixin):
    """A convenience wrapper for row-wise transformers to apply
    transformation to all rows.

    This estimator allows to create a transformer that works on all rows
    from a passed transformer that works on a
    single row. This is useful for applying transformers to the time-series
    in the rows.

    Parameters
    ----------
    transformer : estimator
        An estimator that can work on a row (i.e. a univariate time-series
        in form of a numpy array or pandas Series.
        must support `fit` and `transform`
    """

    _required_parameters = ["transformer"]

    def __init__(self, transformer):
        self.transformer = transformer
        super(RowTransformer, self).__init__()

    def transform(self, X, y=None):
        """Apply the `fit_transform()` method of the transformer on each row.
        """
        func = self.transformer.fit_transform
        return self._apply_rowwise(func, X, y)

    def inverse_transform(self, X, y=None):
        """Apply the `fit_transform()` method of the transformer on each row.
        """
        if not hasattr(self.transformer, "inverse_transform"):
            raise AttributeError(
                "Transformer does not have an inverse transform method")
        func = self.transformer.inverse_transform
        return self._apply_rowwise(func, X, y)

    def _apply_rowwise(self, func, X, y=None):
        """Helper function to apply transform or inverse_transform function
        on each row of data container"""
        #self.check_is_fitted()
        X = check_X(X)

        # 1st attempt: apply, relatively fast but not robust
        # try and except, but sometimes breaks in other cases than excepted
        # ValueError
        # Works on single column, but on multiple columns only if columns
        # have equal-length series.
        # try:
        #     Xt = X.apply(self.transformer.fit_transform)
        #
        # # Otherwise call apply on each column separately.
        # except ValueError as e:
        #     if str(e) == "arrays must all be same length":
        #         Xt = pd.concat([pd.Series(col.apply(
        #         self.transformer.fit_transform)) for _, col in X.items()],
        #         axis=1)
        #     else:
        #         raise

        # 2nd attempt: apply but iterate over columns, still relatively fast
        # but still not very robust
        # but column is not 2d and thus breaks if transformer expects 2d input
        try:
            Xt = pd.concat([pd.Series(col.apply(func))
                            for _, col in X.items()], axis=1)

        # 3rd attempt: explicit for-loops, most robust but very slow
        except Exception:
            cols_t = []
            for c in range(X.shape[1]):  # loop over columns
                col = X.iloc[:, c]
                rows_t = []
                for row in col:  # loop over rows in each column
                    row_2d = pd.DataFrame(row)  # convert into 2d dataframe
                    row_t = func(row_2d).ravel()  # apply transform
                    rows_t.append(row_t)  # append transformed rows
                cols_t.append(rows_t)  # append transformed columns

            # if series-to-series transform, flatten transformed series
            Xt = _concat_nested_arrays(
                cols_t)  # concatenate transformed columns

            # tabularise/unnest series-to-primitive transforms
            xt = Xt.iloc[0, 0]
            if isinstance(xt, (pd.Series, np.ndarray)) and len(xt) == 1:
                Xt = from_nested_to_2d_array(Xt)
        return Xt


# load_from_arff_to_dataframe
def load_from_arff_to_dataframe(
    full_file_path_and_name,
    has_class_labels=True,
    return_separate_X_and_y=True,
    replace_missing_vals_with="NaN",
):
    '''
    load time series classification data
    
    Args:
        full_file_path_and_name:
        has_class_labels:
        return_separate_X_and_y:
        replace_missing_vals_with:
    
    Returns:
        dataframe X and labels
    '''
    instance_list = []
    class_val_list = []

    data_started = False
    is_multi_variate = False
    is_first_case = True

    with open(full_file_path_and_name, "r") as f:
        for line in f:

            if line.strip():
                if (
                    is_multi_variate is False
                    and "@attribute" in line.lower()
                    and "relational" in line.lower()
                ):
                    is_multi_variate = True

                if "@data" in line.lower():
                    data_started = True
                    continue

                # if the 'data tag has been found, the header information
                # has been cleared and now data can be loaded
                if data_started:
                    line = line.replace("?", replace_missing_vals_with)

                    if is_multi_variate:
                        if has_class_labels:
                            line, class_val = line.split("',")
                            class_val_list.append(class_val.strip())
                        dimensions = line.split("\\n")
                        dimensions[0] = dimensions[0].replace("'", "")

                        if is_first_case:
                            for _d in range(len(dimensions)):
                                instance_list.append([])
                            is_first_case = False

                        for dim in range(len(dimensions)):
                            instance_list[dim].append(
                                pd.Series(
                                    [float(i) for i in dimensions[dim].split(",")]
                                )
                            )

                    else:
                        if is_first_case:
                            instance_list.append([])
                            is_first_case = False

                        line_parts = line.split(",")
                        if has_class_labels:
                            instance_list[0].append(
                                pd.Series(
                                    [
                                        float(i)
                                        for i in line_parts[: len(line_parts) - 1]
                                    ]
                                )
                            )
                            class_val_list.append(line_parts[-1].strip())
                        else:
                            instance_list[0].append(
                                pd.Series(
                                    [float(i) for i in line_parts[: len(line_parts)]]
                                )
                            )

    x_data = pd.DataFrame(dtype=np.float32)
    for dim in range(len(instance_list)):
        x_data["dim_" + str(dim)] = instance_list[dim]

    if has_class_labels:
        if return_separate_X_and_y:
            return x_data, np.asarray(class_val_list)
        else:
            x_data["class_vals"] = pd.Series(class_val_list)

    return x_data


# Feature Extraction ARFIMA Models
def arfima_coefs(x, max_pq=5):
    x = np.asarray(x).ravel()
    model = forecast.arfima(vectors.FloatVector(x), **{'estim':'ls', 'max.p':max_pq, 'max.q':max_pq})

    feature = np.zeros((2, max_pq))
    lists = [np.asarray(model[4]), np.asarray(model[5])]
    for i, sub in enumerate(lists):
        feature[i][0:len(sub)] = sub

    return np.concatenate((np.asarray(model[3]), feature.flatten())).ravel()


# Additional Feature Extraction
def other_features(x):
    x = np.asarray(x).ravel()
    n = max(20, min(int(len(x)*0.50), 100))
    add1 = np.asarray([fc.maximum(x), fc.minimum(x), fc.mean(x), fc.median(x), fc.mean_abs_change(x),
                       fc.mean_change(x), fc.mean_second_derivative_central(x), fc.percentage_of_reoccurring_datapoints_to_all_datapoints(x),
                       fc.percentage_of_reoccurring_values_to_all_values(x), fc.ratio_value_number_to_time_series_length(x),
                       fc.root_mean_square(x), fc.sample_entropy(x), fc.skewness(x), fc.standard_deviation(x), 
                       fc.sum_of_reoccurring_values(x), fc.sum_of_reoccurring_data_points(x), fc.sum_values(x), fc.variance(x), 
                       fc.variance_larger_than_standard_deviation(x), fc.variation_coefficient(x), fc.abs_energy(x), fc.absolute_maximum(x), 
                       fc.absolute_sum_of_changes(x), fc.benford_correlation(x), fc.count_above_mean(x), fc.first_location_of_maximum(x), 
                       fc.first_location_of_minimum(x), fc.has_duplicate(x), fc.has_duplicate_max(x), fc.has_duplicate_min(x), 
                       fc.kurtosis(x), fc.last_location_of_maximum(x), fc.last_location_of_minimum(x),fc.length(x), 
                       fc.longest_strike_above_mean(x), fc.longest_strike_below_mean(x), fc.count_above(x, 0), fc.count_below(x, 0)]).ravel()
    
    add2 = np.asarray([fc.time_reversal_asymmetry_statistic(x, i) for i in range(1, n, 1)] + #
                      [fc.c3(x, i) for i in range(1, n, 1)] + [fc.cid_ce(x, i) for i in [True, False]] + #
                      [fc.symmetry_looking(x, [{'r': i}])[0][1] for i in np.linspace(0.0, 1.0, n)] + # 
                      [fc.large_standard_deviation(x, i) for i in np.linspace(0.05, 1.0, 30)] +
                      [fc.quantile(x, i) for i in np.linspace(0.1, 1.0, 10)] + [fc.number_peaks(x, i) for i in range(1, n, 1)] + #
                      [fc.agg_autocorrelation(x, [{"f_agg": i, "maxlag": n}])[0][1] for i in ['mean', 'var', 'std', 'median']] +
                      [fc.autocorrelation(x, i) for i in range(1, n, 1)] + [fc.partial_autocorrelation(x, [{'lag': i}])[0][1] for i in range(1, n, 1)] + #
                      [fc.binned_entropy(x, i) for i in np.arange(5,n,5)] + [fc.number_cwt_peaks(x, i) for i in range(1, n, 1)] +  #
                      [fc.index_mass_quantile(x,[{'q': i}])[0][1] for i in np.linspace(0.1, 1.0, 10)] +
                      [tuple(fc.cwt_coefficients(x, [{'widths':[2,5,10,20], 'coeff': j, 'w': k}]))[0][1] for j, k in itertools.product(np.arange(0, n, 1),
                                                                                                                                       [2,5,10,20])] + #
                      [tuple(fc.spkt_welch_density(x, [{"coeff": i}]))[0][1] for i in range(2, n, 1)] +
                      [fc.change_quantiles(x, j, i, k, l) for i, j, k, l in itertools.product(np.linspace(0.0, 1.0, 11), 
                                                                                              np.linspace(0.0, 1.0, 11), 
                                                                                              [True, False], 
                                                                                              ['mean', 'var', 'std', 'median']) if i > j] +
                      [tuple(fc.fft_coefficient(x, [{'coeff': i, 'attr': j}]))[0][1] for i, j in itertools.product(np.arange(0, n, 1), 
                                                                                                                   ['real', 'imag', 'abs', 'angle'])] + #
                      [tuple(fc.fft_aggregated(x, [{'aggtype': s}]))[0][1] for s in ['centroid', 'variance', 'skew', 'kurtosis']] +
                      [fc.value_count(x, i) for i in [0, 1, -1]] + [fc.friedrich_coefficients(x, [{"m": 3, "r": 30, "coeff": i}])[0][1] for i in [0,1,2,3]] +
                      [fc.linear_trend(x, [{"attr": i}])[0][1] for i in ['pvalue', 'rvalue', 'intercept', 'slope', 'stderr']] +
                      [tuple(fc.agg_linear_trend(x, [{'attr': i, 'chunk_len': l, 'f_agg': f}]))[0][1] for i, l, f in itertools.product(['pvalue', 'rvalue', 'intercept', 'slope', 'stderr'], 
                                                                                                                                       np.arange(5, n, 5), 
                                                                                                                                       ['max', 'min', 'mean', 'var'])] + #
                      [fc.energy_ratio_by_chunks(x, [{'num_segments': 10, 'segment_focus': i}])[0][1] for i in np.arange(0, 10, 1)] +
                      [fc.number_crossing_m(x, i) for i in [-1,0,1]] + [fc.ratio_beyond_r_sigma(x, i) for i in np.arange(0.25, 10, 0.25)] +
                      [fc.lempel_ziv_complexity(x, i) for i in np.arange(2, n, 2)] + #
                      [fc.fourier_entropy(x, i) for i in np.arange(2, n, 2)] + #
                      [fc.permutation_entropy(x, 1, i) for i in np.arange(3, n, 1)] #
                    )
    add12 = np.concatenate((add1, add2)).ravel()

    return np.nan_to_num(add12, nan=0, neginf=0, posinf=0) # np.where(np.isnan(add12) or , 0, add12).ravel() 
