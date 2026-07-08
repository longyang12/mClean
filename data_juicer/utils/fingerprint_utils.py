import functools
import inspect
from pathlib import Path
from typing import Any, Dict, List, Union

import dill
import numpy as np
import xxhash
from datasets.fingerprint import (
    _CACHING_ENABLED,
    fingerprint_warnings,
    format_kwargs_for_fingerprint,
    format_transform_for_fingerprint,
    generate_random_fingerprint,
    validate_fingerprint,
)
from loguru import logger


FINGERPRINT_OWNER_ATTR_EXCLUDES = {
    "work_dir",
    "skip_op_error",
    "batch_size",
    "num_proc",
    "cpu_required",
    "gpu_required",
    "mem_required",
    "num_cpus",
    "num_gpus",
    "memory",
    "runtime_env",
    "ray_execution_mode",
    "model_key",
}


def normalize_fingerprint_value(value):
    """
    Convert common Python objects into a stable structure suitable for hashing.
    We intentionally avoid serializing arbitrary runtime objects here.
    """

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, bytes):
        return {"__type__": "bytes", "hex": value.hex()}

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {str(k): normalize_fingerprint_value(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}

    if isinstance(value, (list, tuple)):
        return [normalize_fingerprint_value(v) for v in value]

    if isinstance(value, (set, frozenset)):
        return sorted(normalize_fingerprint_value(v) for v in value)

    if callable(value):
        return normalize_function_identity(value)

    return {
        "__type__": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
        "repr": repr(value),
    }


def extract_owner_semantic_state(owner):
    """
    Extract stable, semantically relevant operator configuration from a bound
    method owner while excluding runtime-only attributes.
    """

    if owner is None:
        return None

    if hasattr(owner, "_init_parameters") and isinstance(owner._init_parameters, dict):
        return normalize_fingerprint_value(owner._init_parameters)

    owner_state = {}
    for key, value in sorted(owner.__dict__.items()):
        if key.startswith("_") or key in FINGERPRINT_OWNER_ATTR_EXCLUDES:
            continue
        if inspect.ismethod(value) or inspect.isfunction(value):
            continue
        owner_state[key] = normalize_fingerprint_value(value)

    return owner_state


class Hasher:
    """Hasher that accepts python objects as inputs."""

    dispatch: Dict = {}

    def __init__(self):
        self.m = xxhash.xxh64()

    @classmethod
    def hash_bytes(cls, value: Union[bytes, List[bytes]]) -> str:
        value = [value] if isinstance(value, bytes) else value
        m = xxhash.xxh64()
        for x in value:
            m.update(x)
        return m.hexdigest()

    @classmethod
    def hash_default(cls, value: Any) -> str:
        """
        Use dill to serialize objects to avoid serialization failures.
        """
        return cls.hash_bytes(dill.dumps(value))

    @classmethod
    def hash(cls, value: Any) -> str:
        if type(value) in cls.dispatch:
            return cls.dispatch[type(value)](cls, value)
        else:
            return cls.hash_default(value)

    def update(self, value: Any) -> None:
        header_for_update = f"=={type(value)}=="
        value_for_update = self.hash(value)
        self.m.update(header_for_update.encode("utf8"))
        self.m.update(value_for_update.encode("utf-8"))

    def hexdigest(self) -> str:
        return self.m.hexdigest()


def update_fingerprint(fingerprint, transform, transform_args):
    """
    Combining various objects to update the fingerprint.
    """

    hasher = Hasher()
    hasher.update(fingerprint)
    try:
        hasher.update(transform)
    except:  # noqa various errors might raise here from pickle or dill
        if _CACHING_ENABLED:
            if not fingerprint_warnings.get("update_fingerprint_transform_hash_failed", False):
                logger.warning(
                    f"Transform {transform} couldn't be hashed properly, \
                     a random hash was used instead. Make sure your \
                     transforms and parameters are serializable with \
                     pickle or dill for the dataset fingerprinting and \
                     caching to work. If you reuse this transform, the \
                     caching mechanism will consider it to be different \
                     from the previous calls and recompute everything. \
                     This warning is only showed once. Subsequent hashing \
                     failures won't be showed."
                )
                fingerprint_warnings["update_fingerprint_transform_hash_failed"] = True
            else:
                logger.info(
                    f"Transform {transform} couldn't be hashed properly, \
                     a random hash was used instead."
                )
        else:
            logger.info(
                f"Transform {transform} couldn't be hashed properly, a \
                 random hash was used instead. This doesn't affect caching \
                 since it's disabled."
            )

        return generate_random_fingerprint()
    for key in sorted(transform_args):
        hasher.update(key)
        try:
            hasher.update(transform_args[key])
        except:  # noqa various errors might raise here from pickle or dill
            if _CACHING_ENABLED:
                if not fingerprint_warnings.get("update_fingerprint_transform_hash_failed", False):
                    logger.warning(
                        f"Parameter '{key}'={transform_args[key]} of the \
                         transform {transform} couldn't be hashed properly, \
                         a random hash was used instead. Make sure your \
                         transforms and parameters are serializable with \
                         pickle or dill for the dataset fingerprinting and \
                         caching to work. If you reuse this transform, the \
                         caching mechanism will consider it to be different \
                         from the previous calls and recompute everything. \
                         This warning is only showed once. Subsequent hashing \
                         failures won't be showed."
                    )
                    fingerprint_warnings["update_fingerprint_transform_hash_failed"] = True
                else:
                    logger.info(
                        f"Parameter '{key}'={transform_args[key]} of the \
                         transform {transform} couldn't be hashed properly, \
                         a random hash was used instead."
                    )
            else:
                logger.info(
                    f"Parameter '{key}'={transform_args[key]} of the transform \
                     {transform} couldn't be hashed properly, a random hash \
                     was used instead. This doesn't affect caching since it's \
                     disabled."
                )
            return generate_random_fingerprint()
    return hasher.hexdigest()


def generate_stage_fingerprint(dataset_fingerprint, op_name, stage_name, stage_signature):
    """
    Generate a stable fingerprint for an explicitly decoupled internal stage.
    """

    transform = {
        "type": "decoupled_filter_stage",
        "op_name": op_name,
        "stage": stage_name,
    }
    transform_args = {
        "stage_signature": normalize_fingerprint_value(stage_signature),
    }
    new_fingerprint = update_fingerprint(dataset_fingerprint, transform, transform_args)
    validate_fingerprint(new_fingerprint)
    return new_fingerprint


def normalize_function_identity(function):
    """
    Convert a callable object into a stable, serializable identity for
    fingerprinting. This avoids cross-run and cross-platform drift caused by
    serializing wrapper/closure/partial objects directly.
    """

    if function is None:
        return None

    raw_function = function
    partial_chain = []
    while isinstance(raw_function, functools.partial):
        partial_chain.append(
            {
                "args": list(raw_function.args),
                "keywords": dict(raw_function.keywords or {}),
            }
        )
        raw_function = raw_function.func

    unwrap_depth = 0
    while hasattr(raw_function, "__wrapped__"):
        raw_function = raw_function.__wrapped__
        unwrap_depth += 1

    owner = getattr(raw_function, "__self__", None)
    owner_name = getattr(owner, "_name", None)
    if owner_name is None and owner is not None:
        owner_name = owner.__class__.__name__

    qualname = getattr(raw_function, "__qualname__", None)
    if qualname is None:
        qualname = getattr(raw_function, "__name__", raw_function.__class__.__name__)

    function_identity = {
        "module": getattr(raw_function, "__module__", ""),
        "qualname": qualname,
        "owner": owner_name,
        "unwrap_depth": unwrap_depth,
    }

    # Keep partial call-time arguments as part of the semantic identity.
    if partial_chain:
        function_identity["partial_chain"] = partial_chain

    # Local functions are still representable via qualname; record their code
    # location explicitly to reduce ambiguity without serializing the whole object.
    code = getattr(raw_function, "__code__", None)
    if inspect.isfunction(raw_function) and code is not None:
        function_identity["code"] = {
            "filename": code.co_filename,
            "firstlineno": code.co_firstlineno,
        }

    owner_state = extract_owner_semantic_state(owner)
    if owner_state:
        function_identity["owner_state"] = owner_state

    return function_identity


def generate_fingerprint(ds, *args, **kwargs):
    """
    Generate new fingerprints by using various kwargs of the dataset.
    """
    if args:
        args = list(args)
        dataset_kwargs = {"shard": ds, "function": normalize_function_identity(args[0])}
    else:
        dataset_kwargs = {"shard": ds}
        if "function" in kwargs and kwargs["function"] is not None:
            kwargs = dict(kwargs)
            kwargs["function"] = normalize_function_identity(kwargs["function"])
    dataset_kwargs.update(kwargs)

    # we create a unique hash from the function,
    # current dataset file and the mapping args
    transform = format_transform_for_fingerprint(ds._map_single)
    kwargs_for_fingerprint = format_kwargs_for_fingerprint(ds._map_single, (), dataset_kwargs)
    kwargs_for_fingerprint["fingerprint_name"] = "new_fingerprint"
    new_fingerprint = update_fingerprint(ds._fingerprint, transform, kwargs_for_fingerprint)
    validate_fingerprint(new_fingerprint)
    return new_fingerprint
