"""Subband reshaping for spectral convolutional models."""


def cac2cws(x, num_subbands):
    b, c, f, t = x.shape
    return x.reshape(b, c * num_subbands, f // num_subbands, t)


def cws2cac(x, num_subbands):
    b, c, f, t = x.shape
    return x.reshape(b, c // num_subbands, f * num_subbands, t)
