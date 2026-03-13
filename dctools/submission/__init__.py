"""Model output submission and validation system.

Provides a user-friendly interface for participants to submit predictions
to the Data Challenge benchmark, validate data conformance, and launch
the evaluation pipeline.

Inspired by benchmarking best practices from WeatherBench2 and OceanBench.
"""

from dctools.submission.validator import InputInfo, SubmissionValidator
from dctools.submission.submission import ModelSubmission

__all__ = ["InputInfo", "SubmissionValidator", "ModelSubmission"]
