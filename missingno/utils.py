"""Utility functions for missingno."""

import numpy as np
import warnings


def nullity_sort(df, sort=None, axis="columns"):
    """
    Sorts a DataFrame according to its nullity, in either ascending or descending order.

    :param df: The DataFrame object being sorted.
    :param sort: The sorting method: either "ascending", "descending", or None (default).
    :param axis: Axis along which to sort. May be ``"index"`` or ``"columns"``.
        ``"rows"`` is supported for backwards compatibility but is deprecated.
    :return: The nullity-sorted DataFrame.
    """
    if sort is None:
        return df
    elif sort not in ["ascending", "descending"]:
        raise ValueError(
            'The "sort" parameter must be set to "ascending" or "descending".'
        )

    # Older versions of missingno accepted "rows" as an alias for the index axis.
    if axis == "rows":
        warnings.warn(
            'Using axis="rows" is deprecated and will be removed in a future '
            'release. Use axis="index" instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        axis = "index"

    if axis not in ["index", "columns"]:
        raise ValueError('The "axis" parameter must be set to "index" or "columns".')

    if axis == "columns":
        if sort == "ascending":
            return df.iloc[np.argsort(df.count(axis="columns").values), :]
        elif sort == "descending":
            return df.iloc[np.flipud(np.argsort(df.count(axis="columns").values)), :]
    elif axis == "index":
        if sort == "ascending":
            return df.iloc[:, np.argsort(df.count(axis="index").values)]
        elif sort == "descending":
            return df.iloc[:, np.flipud(np.argsort(df.count(axis="index").values))]


def nullity_filter(df, filter=None, p=0, n=0):
    """
    Filters a DataFrame according to its nullity, using some combination of 'top' and 'bottom' numerical and
    percentage values. Percentages and numerical thresholds can be specified simultaneously: for example,
    to get a DataFrame with columns of at least 75% completeness but with no more than 5 columns, use
    `nullity_filter(df, filter='top', p=.75, n=5)`.

    :param df: The DataFrame whose columns are being filtered.
    :param filter: The orientation of the filter being applied to the DataFrame. One of, "top", "bottom",
    or None (default). The filter will simply return the DataFrame if you leave the filter argument unspecified or
    as None.
    :param p: A completeness ratio cut-off. If non-zero the filter will limit the DataFrame to columns with at least p
    completeness. Input should be in the range [0, 1].
    :param n: A numerical cut-off. If non-zero no more than this number of columns will be returned.
    :return: The nullity-filtered `DataFrame`.
    """
    if filter == "top":
        if p:
            df = df.iloc[:, [c >= p for c in df.count(axis="index").values / len(df)]]
        if n:
            df = df.iloc[:, np.sort(np.argsort(df.count(axis="index").values)[-n:])]
    elif filter == "bottom":
        if p:
            df = df.iloc[:, [c <= p for c in df.count(axis="index").values / len(df)]]
        if n:
            df = df.iloc[:, np.sort(np.argsort(df.count(axis="index").values)[:n])]
    return df
