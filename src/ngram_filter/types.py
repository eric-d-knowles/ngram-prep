from typing import Optional, Protocol

class ProcessorProtocol(Protocol):
    def __call__(self, token: str) -> Optional[str]: ...
