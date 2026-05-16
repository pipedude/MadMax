class MadMaxError(Exception):
    pass


class ConfigurationError(MadMaxError):
    pass


class WakeWordModelError(MadMaxError):
    pass


class AudioInitializationError(MadMaxError):
    pass


class RealtimeAPIError(MadMaxError):
    pass


class RecoverableConnectionError(RealtimeAPIError):
    pass


class FatalAPIError(RealtimeAPIError):
    pass


class ToolExecutionError(MadMaxError):
    pass


class InactivityTimeoutError(RealtimeAPIError):
    pass
