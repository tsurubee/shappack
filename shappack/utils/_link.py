import numpy as np


class IdentityLink:
    def __str__(self):
        return "identity"

    @staticmethod
    def f(x):
        return x

    @staticmethod
    def finv(x):
        return x


class LogitLink:
    def __str__(self):
        return "logit"

    @staticmethod
    def f(x):
        return np.log(x / (1 - x))

    @staticmethod
    def finv(x):
        return 1 / (1 + np.exp(-x))


def convert_to_link(link):
    if link == "identity":
        return IdentityLink()
    elif link == "logit":
        return LogitLink()
    else:
        raise ValueError("The link argument must be identity or logit.")
