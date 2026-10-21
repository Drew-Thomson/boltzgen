import logging
import warnings


def quiet_startup() -> None:
    """Filter verbose third-party warnings and configure logging."""
    warnings.filterwarnings("ignore", message=r".*predict_dataloader.*num_workers.*")
    warnings.filterwarnings(
        "ignore", message=r".*tensorboardX.*removed as a dependency.*"
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The pynvml package is deprecated",
        category=FutureWarning,
        module=r"torch\.cuda",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Pandas requires version .* or newer of 'numexpr'",
        category=UserWarning,
    )

    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
