"""autopath.utils — shared utilities for pipeline functions and datablocks."""
import functools
import inspect


# Parameters that are operational overrides, never part of a logical call tag.
_TAG_SKIP_DEFAULTS = frozenset({'tag', 'root'})


def _make_tag(func: callable, sig: inspect.Signature, arguments: dict,
              skip: frozenset) -> str:
    """Build a human-readable call string from bound + defaulted arguments.

    Only non-default values (and required positional args) are included, so
    the result reads like the minimal call a user would type.

    Example:
        "autopath.gigaq.pipelines.gigapath_bipolar_feature_bag_clip('CPTAC_206020', single=1)"
    """
    pos_args, kw_args = [], []
    for name, param in sig.parameters.items():
        if name in skip:
            continue
        value = arguments[name]
        is_required = param.default is inspect.Parameter.empty
        is_positional = param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        if is_required:
            if is_positional:
                pos_args.append(repr(value))
            else:
                kw_args.append(f"{name}={repr(value)}")
        elif value != param.default:
            kw_args.append(f"{name}={repr(value)}")

    qualname = f"{func.__module__}.{func.__qualname__}"
    return f"{qualname}({', '.join(pos_args + kw_args)})"


def tagged(func=None, *, skip: frozenset = _TAG_SKIP_DEFAULTS):
    """Decorator for pipeline functions that auto-computes a call-string tag.

    When the decorated function is called with ``tag=None`` (or tag is omitted),
    the decorator synthesises a tag of the form::

        "autopath.gigaq.pipelines.gigapath_bipolar_feature_bag_clip('CPTAC_206020', single=1)"

    showing only the arguments that differ from their defaults.  If ``tag`` is
    supplied explicitly (including by an upstream pipeline that already has a
    tag), it is passed through unchanged.

    The decorated function receives ``tag`` as a normal keyword argument and
    need not know whether it was supplied by the caller or synthesised here.

    Usage::

        @tagged
        def my_pipeline(name, *, tag=None, root=None, n_workers=1):
            clip = MyClip(...)
            clip.tag = tag   # propagate down to the datablock
            return clip

    Parameters
    ----------
    skip : frozenset
        Parameter names to exclude from the tag string.  Defaults to
        ``{'tag', 'root'}`` — operational overrides that are not part of a
        pipeline's logical identity.
    """
    # Allow both @tagged and @tagged(skip={...}) usage
    if func is None:
        return functools.partial(tagged, skip=skip)

    sig = inspect.signature(func)
    # Validate that the function actually accepts 'tag'
    if 'tag' not in sig.parameters:
        raise TypeError(
            f"@tagged: {func.__qualname__} must have a 'tag' parameter "
            f"(e.g. tag: str | None = None)"
        )

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        if bound.arguments.get('tag') is None:
            bound.arguments['tag'] = _make_tag(func, sig, bound.arguments, skip)
        return func(*bound.args, **bound.kwargs)

    return wrapper
