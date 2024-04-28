import logging

import polars as pl

logger = logging.getLogger(__name__)


def _unnest_all(schema, separator):
    def _unnest(schema, path=[]):
        for name, dtype in schema.items():
            base_type = dtype.base_type()

            if base_type == pl.Struct:
                yield from _unnest(dtype.to_schema(), path + [name])
            else:
                yield path + [name], dtype

    for (col, *fields), dtype in _unnest(schema):
        expr = pl.col(col)

        for field in fields:
            expr = expr.struct[field]

        not_empty_fields = [f for f in fields if f.strip() != ""]
        if len(not_empty_fields) > 0:
            name = separator.join([col] + not_empty_fields)
        else:
            name = col

        yield expr.alias(name)


def unnest_all(df: pl.DataFrame, separator=".") -> pl.DataFrame:
    return df.select(_unnest_all(df.schema, separator))


def read_trace_file(file_path: str) -> pl.DataFrame:
    """Reads the trace file and returns the trace instance."""
    events = pl.read_ndjson(file_path)
    return Trace(unnest_all(events))


class Trace:
    def __init__(self, events: pl.DataFrame):
        self.events = events
        if isinstance(events, (list, dict)):
            self.events = pl.DataFrame(events)
            self.events = unnest_all(self.events)

        # We may want to sort the events by time as now we have multiple traces from different processes

        # # events shouldn't have any columns that are nested dictionaries
        # for col in self.events.columns:
        #     if any([isinstance(e, dict) for e in self.events[col]]):
        #         print(self.events[col].describe())
        #         raise ValueError(f"Column {col} contains nested lists or dictionaries. Please flatten the trace before creating Trace instance.")

    def filter(self, predicate):
        # TODO: need to think about how to implement this, as pre-conditions for bugs like DS-1801 needs to take multiple events into account
        raise NotImplementedError("filter method is not implemented yet.")
        return Trace(self.events[self.events.apply(predicate, axis=1)])

    def find_variables_states(self) -> list:
        """Find all variables and their states from the trace."""
        # the

    def scan_to_groups(self, param_selectors: list):
        """Extract from trace, groups of events

        args:
            param_selectors: list
                A list of functions that take an event as input and return a value retrieved from the event.
                If a value is not found, the function should return None. The length of the list should be equal to
                the number of variables in the relation.
        """

        # raise NotImplementedError("group method is not implemented for pandas yet.")

        groups: list[list[object]] = []
        group: list[object] = []
        param_selector_itr = iter(param_selectors)
        curr_param_selector = next(param_selector_itr, None)
        assert curr_param_selector is not None, "param_selectors should not be empty"

        for e in self.events:
            if curr_param_selector is None:
                groups.append(group)
                group = []
                param_selector_itr = iter(param_selectors)
                curr_param_selector = next(param_selector_itr, None)
            elif not callable(curr_param_selector):
                # curr_param_selector is a value
                group.append(curr_param_selector)
            elif curr_param_selector(e):
                group.append(curr_param_selector(e))
                curr_param_selector = next(param_selector_itr, None)

        # dealing with the left-over group
        if len(group) == len(param_selectors):
            groups.append(group)
        elif len(group) > 0:
            logger.debug(
                f"Left-over group: {group} not added to groups as it does not have enough elements."
            )

        return groups
