"""Configuration file reader for TXT format files."""

import os
from typing import Dict, Any, Optional, Tuple, Union, List


class ConfigReader:
    """
    @class ConfigReader
    @brief Class for reading and parsing TXT configuration files.

    This class reads a configuration file containing key=value pairs
    and automatically converts values into appropriate Python types:
    - strings
    - integers and floats
    - booleans
    - lists
    - tuples
    - None/null

    Keys are internally stored in uppercase format.
    """

    def __init__(self, filename: str) -> None:
        """
        @brief Constructor of the ConfigReader class.

        Initializes the configuration reader, verifies that the file exists,
        and loads its content into the internal dictionary.

        @param filename Path of the configuration file to read.
        """

        ## @brief Name of the configuration file.
        self.filename = filename

        ## @brief Dictionary containing parameters read from the file.
        self.config_dict: Dict[str, Any] = {}

        self._validate_file_exists()
        self._read_configuration_file()

    def _validate_file_exists(self) -> None:
        """
        @brief Checks that the configuration file exists.

        @exception FileNotFoundError Raised if the file does not exist.
        """
        if not os.path.exists(self.filename):
            raise FileNotFoundError(
                f"Configuration file '{self.filename}' not found. "
                f"Please ensure the file exists at the specified path."
            )

    def _read_configuration_file(self) -> None:
        """
        @brief Reads the configuration file line by line.

        Each line is passed to the parsing method to extract
        the key-value pair.

        @exception ValueError Raised if the file cannot be read.
        """
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
        """
        @brief Processes a single line of the configuration file.

        Empty or commented lines are ignored. All other lines are
        interpreted as key=value pairs.

        @param line Line from the configuration file.
        @param line_number Line number in the file.
        """
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
        """
        @brief Parses a line containing a key=value pair.

        @param line Line to analyze.
        @param line_number Line number in the file.
        @return Tuple containing the key and the converted value.

        @exception ValueError Raised if the line format is invalid.
        """
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
        """
        @brief Converts a string into the appropriate Python type.

        The method attempts to interpret the value as:
        - quoted string
        - list
        - tuple
        - boolean
        - None/null
        - integer or float
        - plain string

        @param value_string Value in string format.
        @return Value converted to the correct Python type.
        """
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
        """
        @brief Checks whether a string is enclosed in quotes.

        @param value_string String to check.
        @return True if the string is quoted, otherwise False.
        """
        if len(value_string) < 2:
            return False

        return (
            (value_string[0] == '"' and value_string[-1] == '"') or
            (value_string[0] == "'" and value_string[-1] == "'")
        )

    def _parse_list_value(self, list_string: str) -> List[Any]:
        """
        @brief Converts a string representing a list.

        @param list_string String containing a list in the format [a, b, c].
        @return Python list with converted values.
        """
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
        """
        @brief Converts a string representing a tuple.

        @param tuple_string String containing a tuple in the format (a, b, c).
        @return Python tuple with converted values.
        """
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
        """
        @brief Converts a numeric string into int or float.

        @param value_string String to convert.
        @return Numeric value or None if conversion fails.
        """
        try:
            if '.' in value_string or 'e' in value_string.lower():
                return float(value_string)
            return int(value_string)
        except ValueError:
            return None

    def get_dict(self) -> Dict[str, Any]:
        """
        @brief Returns a copy of the configuration dictionary.

        @return Dictionary containing all configuration parameters.
        """
        return self.config_dict.copy()

    def get(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        """
        @brief Retrieves the value associated with a configuration key.

        @param key Name of the key.
        @param default Default value if the key does not exist.
        @return Value associated with the key or the default.
        """
        return self.config_dict.get(key.upper(), default)

    def __getitem__(self, key: str) -> Any:
        """
        @brief Allows parameter access using index syntax.

        Example:
        config["PARAMETER"]

        @param key Name of the key.
        @return Value associated with the key.

        @exception KeyError Raised if the key does not exist.
        """
        try:
            return self.config_dict[key.upper()]
        except KeyError as error:
            raise KeyError(
                f"Configuration key '{key}' not found. "
                f"Available keys: {list(self.config_dict.keys())}"
            ) from error

    def __contains__(self, key: str) -> bool:
        """
        @brief Checks whether a key exists in the configuration.

        @param key Name of the key.
        @return True if the key exists.
        """
        return key.upper() in self.config_dict

    def keys(self):
        """
        @brief Returns all configuration keys.
        """
        return self.config_dict.keys()

    def values(self):
        """
        @brief Returns all configuration values.
        """
        return self.config_dict.values()

    def items(self):
        """
        @brief Returns all key-value pairs in the configuration.
        """
        return self.config_dict.items()

    def __str__(self) -> str:
        """
        @brief Returns a readable representation of the configuration.

        @return Formatted string containing all keys and values.
        """
        if not self.config_dict:
            return f"Empty configuration from '{self.filename}'"

        lines = [f"Configuration from '{self.filename}':"]
        for key, value in sorted(self.config_dict.items()):
            value_type = type(value).__name__
            lines.append(f"  {key}: {repr(value)} ({value_type})")

        return '\n'.join(lines)

    def __repr__(self) -> str:
        """
        @brief Technical representation of the object.
        """
        return f"ConfigReader(filename='{self.filename}')"


def read_from_file(filename: str) -> Dict[str, Any]:
    """
    @brief Utility function to directly read a configuration file.

    @param filename Path of the configuration file.
    @return Dictionary containing the parsed parameters.
    """
    reader = ConfigReader(filename)
    return reader.get_dict()
