"""Controlled scientific-domain errors shared across CompoSED components."""


class ModelDomainError(ValueError):
    """Parameter values are finite but describe an invalid physical model.

    This exception is reserved for parameter-dependent support violations, such
    as a galaxy age exceeding the age of the Universe at its redshift. The
    likelihood converts it to ``-inf``. Configuration errors, missing
    parameters, unit mismatches, and shape errors must use their ordinary
    exception types and remain visible to the user.
    """
