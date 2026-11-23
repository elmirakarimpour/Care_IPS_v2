"""Pipeline orchestration is intentionally explicit and not yet end-to-end."""
class PipelineNotReady(RuntimeError):
    pass

def run_experiment(*args, **kwargs):
    raise PipelineNotReady(
        "Training data, validated agent dependencies, constrained reasoning, and full evaluation are required."
    )
