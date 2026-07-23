"""Utilitários numéricos para conversões escalares seguras."""

from __future__ import annotations


def to_float(value: object) -> float:
    """Converte um valor escalar (numpy/pandas/polars) para ``float`` nativo.

    Centraliza a conversão em um único ponto para contornar a tipagem parcial
    de bibliotecas de terceiros (ex.: ``pandas.Series.__getitem__`` e
    ``polars.Series.mean`` não anunciam com precisão o tipo escalar retornado).

    Parameters
    ----------
    value : object
        Valor escalar conversível para ``float`` (ex.: ``numpy.float64``).

    Returns
    -------
    float
        O valor convertido.

    Raises
    ------
    TypeError
        Se o valor não for conversível para ``float`` (ex.: ``None``).

    Examples
    --------
    >>> to_float(0.5)
    0.5
    """
    return float(value)  # type: ignore[arg-type]
