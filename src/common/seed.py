"""Fix all random seeds for reproducibility."""

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        # TensorFlow 2.x
        if hasattr(tf, "random") and hasattr(tf.random, "set_seed"):
            tf.random.set_seed(seed)
        # TensorFlow 1.x fallback
        elif hasattr(tf, "set_random_seed"):
            tf.set_random_seed(seed)  # type: ignore[attr-defined]
    except ImportError:
        # TensorFlow not installed — silently ignore
        pass
    except Exception:
        # Any other TF-related issue should not break the pipeline
        pass
