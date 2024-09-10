import logging
import textwrap


class LoggerWrapper:
    """Wrapper for standard python logger"""
    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger

    def debug(self, *args, **kwargs) -> None:
        if self.logger:
            self.logger.debug(*args, **kwargs)

    def info(self, *args, **kwargs) -> None:
        if self.logger:
            self.logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs) -> None:
        if self.logger:
            self.logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs) -> None:
        if self.logger:
            self.logger.error(*args, **kwargs)

    def critical(self, *args, **kwargs) -> None:
        if self.logger:
            self.logger.critical(*args, **kwargs)

    def exception(self, *args, **kwargs) -> None:
        if self.logger:
            self.logger.exception(*args, **kwargs)

    @classmethod
    def indent_text(cls, text: str | None, amount: int = 4, char: str = " ") -> str:
        """Indent text by amount * char."""
        if text is None:
            return ""

        return textwrap.indent(text, amount * char)
