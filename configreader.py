"""Configuration file reader for TXT format files."""

import os
from typing import Dict, Any, Optional, Tuple, Union, List


class ConfigReader:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.config_dict: Dict[str, Any] = {}

        self._validate_file_exists()
        self._read_configuration_file()

    def _validate_file_exists(self) -> None:
        if not os.path.exists(self.filename):
            raise FileNotFoundError(
                f"Configuration file '{self.filename}' not found. "
                f"Please ensure the file exists at the specified path."
            )

    def _read_configuration_file(self) -> None:
        try:
            with open(self.filename, 'r', encoding='utf-8') as config_file:
                for line_number, line in enumerate(config_file, 1):
                    self._process_configuration_line(line.strip(), line_number)
        except IOError as error:
            raise ValueError(
                f"Failed to read configuration file '{self.filename}': {error}"
            ) from error

    def _process_configuration_line(
        self,
        line: str,
        line_number: int
    ) -> None:
        if not line or line.startswith('#'):
            return
        try:
            key, value = self._parse_key_value_pair(line, line_number)
            self.config_dict[key] = value
        except ValueError as error:
            print(f"Warning (line {line_number}): {error}")

    def _parse_key_value_pair(
        self,
        line: str,
        line_number: int
    ) -> Tuple[str, Any]:
        if '=' not in line:
            raise ValueError(
                f"Invalid format: expected 'key=value', got '{line}'"
            )

        key, raw_value = line.split('=', 1)
        key = key.strip()
        raw_value = raw_value.strip()

        if not key:
            raise ValueError(f"Empty key in line: '{line}'")

        parsed_value = self._parse_configuration_value(raw_value)

        return key.upper(), parsed_value

    def _parse_configuration_value(self, value_string: str) -> Any:
        if self._str_quoted(value_string):
            return value_string[1:-1]

        if value_string.startswith('[') and value_string.endswith(']'):
            return self._parse_list_value(value_string)

        if value_string.startswith('(') and value_string.endswith(')'):
            return self._parse_tuple_value(value_string)

        if value_string.lower() == 'true':
            return True
        elif value_string.lower() == 'false':
            return False
        elif value_string.lower() in ['none', 'null']:
            return None

        numeric_value = self._parse_numeric_value(value_string)
        if numeric_value is not None:
            return numeric_value

        return value_string

    @staticmethod
    def _str_quoted(value_string: str) -> bool:
        if len(value_string) < 2:
            return False

        return (
            (value_string[0] == '"' and value_string[-1] == '"') or
            (value_string[0] == "'" and value_string[-1] == "'")
        )

    def _parse_list_value(self, list_string: str) -> List[Any]:
        inner_content = list_string[1:-1].strip()
        if not inner_content:
            return []

        elements = [
            element.strip()
            for element in inner_content.split(',')
        ]

        return [
            self._parse_configuration_value(element)
            for element in elements
        ]

    def _parse_tuple_value(self, tuple_string: str) -> Tuple[Any, ...]:
        inner_content = tuple_string[1:-1].strip()
        if not inner_content:
            return ()

        elements = [
            element.strip()
            for element in inner_content.split(',')
        ]

        return tuple(
            self._parse_configuration_value(element)
            for element in elements
        )

    @staticmethod
    def _parse_numeric_value(value_string: str) -> Optional[Union[int, float]]:
        try:
            if '.' in value_string or 'e' in value_string.lower():
                return float(value_string)
            return int(value_string)
        except ValueError:
            return None

    def get_dict(self) -> Dict[str, Any]:
        return self.config_dict.copy()

    def get(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        return self.config_dict.get(key.upper(), default)

    def __getitem__(self, key: str) -> Any:
        try:
            return self.config_dict[key.upper()]
        except KeyError as error:
            raise KeyError(
                f"Configuration key '{key}' not found. "
                f"Available keys: {list(self.config_dict.keys())}"
            ) from error

    def __contains__(self, key: str) -> bool:
        return key.upper() in self.config_dict

    def keys(self):
        return self.config_dict.keys()

    def values(self):
        return self.config_dict.values()

    def items(self):
        return self.config_dict.items()

    def __str__(self) -> str:
        if not self.config_dict:
            return f"Empty configuration from '{self.filename}'"

        lines = [f"Configuration from '{self.filename}':"]
        for key, value in sorted(self.config_dict.items()):
            value_type = type(value).__name__
            lines.append(f"  {key}: {repr(value)} ({value_type})")

        return '\n'.join(lines)

    def __repr__(self) -> str:
        return f"ConfigReader(filename='{self.filename}')"


def read_from_file(filename: str) -> Dict[str, Any]:
    reader = ConfigReader(filename)
    return reader.get_dict()

