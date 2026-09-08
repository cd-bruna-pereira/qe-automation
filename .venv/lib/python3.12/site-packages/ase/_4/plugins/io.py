import importlib
import re
from collections.abc import Sequence
from functools import cached_property
from importlib.metadata import EntryPoint
from types import ModuleType
from typing import (
    Any,
    BinaryIO,
    Callable,
    Iterator,
    Literal,
    TextIO,
)

from typing_extensions import Concatenate, Self

from ase.atoms import Atoms
from ase.io.formats import IOFormat

from .plugin import MissingFunction, Plugin


class IOFormatPlugin(Plugin):
    def __init__(
        self,
        name: str,
        *,
        citation: str,
        module: ModuleType | None = None,
        description: str,
        code: str,
        reader: str
        | Callable[[], Callable[..., Atoms | Iterator[Atoms]]]
        | None,
        writer: str
        | Callable[
            [], Callable[Concatenate[str | BinaryIO | TextIO, Atoms, ...], None]
        ]
        | None,
        encoding: str | None = None,
        extensions: list[str] | None = None,
        globs: list[str] | None = None,
        magic: list[str] | None = None,
        magic_regex: bytes | None | None = None,
    ):
        super().__init__(name=name, citation=citation, module=module)

        if not any((reader, writer)):
            raise ValueError('IOFormatPlugin must provide a reader or a writer')

        self.name = name
        self.description = description

        assert len(code) == 2
        assert code[0] in list('+1')
        assert code[1] in list('BFS')
        self.code = code
        self.encoding = encoding

        self._reader = reader
        self._writer = writer

        self.extensions = extensions if extensions else []
        self.globs = globs if globs else []
        self.magic = magic if magic else []
        self.magic_regex = magic_regex

    @cached_property
    def can_read(self) -> bool:
        """Check if plugin implements a reader

        If this is false, read() will raise an error
        """
        return self._reader is not None

    @cached_property
    def can_write(self) -> bool:
        """Check if plugin implements a writer

        If this is false, write() will raise an error
        """
        return self._writer is not None

    def read(
        self, *args: list[Any], **kwargs: dict[str, Any]
    ) -> Atoms | Iterator[Atoms]:
        if not self.can_read:
            raise ValueError('This plugin does not implement a reader')
        assert self._resolved_reader is not None
        return self._resolved_reader(*args, **kwargs)

    def write(
        self,
        filename: str | BinaryIO | TextIO,
        images: Atoms | Sequence[Atoms],
        *args: list[Any],
        **kwargs: dict[str, Any],
    ) -> None:
        if not self.can_write:
            raise ValueError('This plugin does not implement a writer')
        assert self._resolved_writer is not None
        self._resolved_writer(filename, images, *args, **kwargs)

    @cached_property
    def _resolved_reader(self) -> Callable[..., Atoms | Iterator[Atoms]] | None:
        if self._reader is None:
            return None
        return self._resolve(self._reader)  # type: ignore[no-any-return]

    @cached_property
    def _resolved_writer(
        self,
    ) -> (
        Callable[
            Concatenate[str | BinaryIO | TextIO, Atoms | Sequence[Atoms], ...],
            None,
        ]
        | None
    ):
        if self._writer is None:
            return None
        return self._resolve(self._writer)  # type: ignore[no-any-return]

    def match_name(self, basename: str) -> bool:
        from fnmatch import fnmatch

        return any(fnmatch(basename, pattern) for pattern in self.globs)

    def match_magic(self, data: bytes) -> bool:
        if self.magic_regex:
            assert not self.magic, 'Define only one of magic and magic_regex'
            match = re.match(self.magic_regex, data, re.M | re.S)
            return match is not None

        from fnmatch import fnmatchcase

        return any(
            fnmatchcase(data, magic + b'*')  # type: ignore[operator, type-var]
            for magic in self.magic
        )


class LegacyIOFormatPlugin(IOFormatPlugin):
    """Wrapper for old IOFormat class to new plugin system

    Note that __init__ is the same as IOFormatPlugin: this class
    should typically be constructed using the
    .from_legacy_ioformat or .from_legacy_entry_point classmethod.
    """

    @cached_property
    def can_read(self) -> bool:
        """Check if plugin implements a reader

        If this is false, read() will raise an error

        New IOFormatPlugin stores a missing reader as None, but for legacy
        implementations we have to check the module content at this point.
        """
        try:
            self._resolved_reader
        except MissingFunction:
            return False
        return True

    @cached_property
    def can_write(self) -> bool:
        """Check if plugin implements a writer

        If this is false, write() will raise an error
        """
        try:
            self._resolved_writer
        except MissingFunction:
            return False
        return True

    @staticmethod
    def _legacy_function_loader(
        name: str,
        module_name: str,
        prefix: Literal['read_', 'write_'],
        error_message: str,
    ) -> Callable[[], Callable[..., Any]]:
        """Get specially-named function from a module, wrapped in Callable

        This is a compatibility tool for working with the legacy IOFormat

        """

        def func() -> Callable[..., Any]:
            formatname = name.replace('-', '_')
            module = importlib.import_module(module_name)
            legacy_function = getattr(module, prefix + formatname, None)
            if legacy_function is None:
                raise MissingFunction(error_message)
            return legacy_function  # type: ignore[no-any-return]

        return func

    @classmethod
    def from_legacy_io_format(
        cls,
        io_format: IOFormat,
        module: ModuleType | None,
        citation: str = '',
    ) -> Self:

        reader = cls._legacy_function_loader(
            io_format.name,
            io_format.module_name,
            'read_',
            f"Can't read from {io_format.name}-format",
        )
        writer = cls._legacy_function_loader(
            io_format.name,
            io_format.module_name,
            'write_',
            f"Can't write to {io_format.name}-format",
        )

        plugin = cls(
            name=io_format.name,
            module=module,
            citation='Imported from legacy IOFormat',
            description=io_format.description,
            code=io_format.code,
            encoding=io_format.encoding,
            reader=reader,
            writer=writer,
            extensions=io_format.extensions,
            globs=io_format.globs,
            magic=io_format.magic,
            magic_regex=io_format.magic_regex,
        )
        return plugin

    @classmethod
    def from_legacy_entry_point(
        cls,
        entry_point: EntryPoint,
    ) -> Self:
        fmt = entry_point.load()

        reader = cls._legacy_function_loader(
            entry_point.name,
            entry_point.module,
            'read_',
            f"Can't read from {entry_point.name}-format",
        )

        writer = cls._legacy_function_loader(
            entry_point.name,
            entry_point.module,
            'write_',
            f"Can't write to {entry_point.name}-format",
        )

        # Note that the "module" attribute of a Plugin should represent where
        # the *plugin* comes from, rather than its underlying implementation
        #
        # Wrapped legacy plugins are a slight grey area but is seems much more
        # informative to use the external package
        plugin = cls(
            name=entry_point.name,
            module=importlib.import_module(entry_point.module),
            citation='Imported from legacy ExternalIOFormat',
            description=fmt.desc,
            code=fmt.code,
            reader=reader,
            writer=writer,
            extensions=fmt.ext,
            globs=fmt.glob,
            magic=fmt.magic,
            magic_regex=fmt.magic_regex,
        )
        return plugin
