class MLPipelineError(Exception):
    """Base error for ML pipeline"""

    pass


class ConfigError(MLPipelineError):
    """Configuration related errors"""

    pass


class DataError(MLPipelineError):
    """Data loading/processing errors"""

    pass


class ModelError(MLPipelineError):
    """Model related errors"""

    pass


class FeatureProcessorError(MLPipelineError):
    """Feature processing errors"""

    pass
