---
name: datablock_method_ordering
description: Rule for ordering methods in Datablock and Datastack classes.
---

# Datablock Method Ordering

When implementing or modifying any `Datablock` or `Datastack` class (or subclasses like `DatastreamTab` and `DatastreamTable`), enforce the following method order:

1. **Datablock Protocol Methods**: 
   These are the core lifecycle and structural methods for datablocks.
   - `__init__`
   - `__post_init__`
   - `__split__` (if Datastack)
   - `__block__` / `__tab__` (if Datastack)
   - `__build__`
   - `__stack__` (if Datastack)
   - `__read__`
   - `path()`
   - `valid()`
   - `validtopic()`

2. **Properties and Accessors**:
   Public properties and data access methods intended for the user/consumer.
   - e.g., `@property name`, `@property n_blocks`, `__len__`, `__getitem__`, `dataset()`, `data()`

3. **Private and Utility Methods**:
   Internal logic, helpers, and data-fetching mechanisms not part of the standard protocol.
   - e.g., `_source_dataset()`, `_compute_metrics()`, `_is_local_fs`
