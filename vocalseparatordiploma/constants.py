SAMPLE_RATE = 44100
SIGNAL_NORMALIZATION_CONSTANT = 32767


class ImmutableDict(dict):
    """
    Immutable dictionary class for constants.
    """
    def __hash__(self):
        return id(self)

    def _immutable(self, *args, **kws):
        raise TypeError('object is immutable')

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear       = _immutable
    update      = _immutable
    setdefault  = _immutable # type: ignore
    pop         = _immutable
    popitem     = _immutable


STFT_DEFAULT_PARAMETERS = ImmutableDict({
    "fs": SAMPLE_RATE,
    "nperseg": 1024,
    "noverlap": 512,
    "window": "hamming"
    })
