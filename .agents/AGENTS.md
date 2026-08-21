# Coding and Documentation Rules

- **No Sphinx/RST Role Markup in Docstrings**: Do not use `:class:`, `:meth:`, `:func:`, `:module:`, or other Sphinx reStructuredText role markup in docstrings. Use plain text or markdown backticks (`ClassName`, `method_name()`) instead. Documentation generation is not currently used.
- **No Defaults for Required Dataclass Parameters**: NEVER add default values (or optional `None` defaults) or alias parameters to required dataclass or `VAR` parameters to accommodate caller keyword mismatches. Always fix the callsite / caller arguments instead of expanding or altering `VAR` dataclass definitions.

