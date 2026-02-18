# SPDX-FileCopyrightText: 2024-present Alan Meeson <am@carefullycalculated.co.uk>
#
# SPDX-License-Identifier: Apache-2.0

# Note: S608 warning for SQL injection vector is disabled, as SQL query construction is necessary for building the
# queries for the filters.
# TODO: find a better way of doing this that doesn't require string construction.
import json
from datetime import datetime
from itertools import chain
from typing import Any, Dict, List

from haystack.errors import FilterError

NO_VALUE = "no_value"


def _convert_filters_to_where_clause_and_params(filters: Dict[str, Any]) -> tuple[str, tuple]:
    """
    Convert Haystack filters to a WHERE clause and a tuple of params to query PostgreSQL.
    """
    if "field" in filters:
        query, values = _parse_comparison_condition(filters)
    else:
        query, values = _parse_logical_condition(filters)

    params = tuple(value for value in values if value != NO_VALUE)

    return query, params


def _parse_logical_condition(condition: Dict[str, Any]) -> tuple[str, List[Any]]:
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "conditions" not in condition:
        msg = f"'conditions' key missing in {condition}"
        raise FilterError(msg)

    operator = condition["operator"]
    if operator not in ["AND", "OR", "NOT"]:
        msg = f"Unknown logical operator '{operator}'. Valid operators are: 'AND', 'OR', 'NOT'"
        raise FilterError(msg)

    # logical conditions can be nested, so we need to parse them recursively
    conditions = []
    for c in condition["conditions"]:
        if "field" in c:
            query, vals = _parse_comparison_condition(c)
        else:
            query, vals = _parse_logical_condition(c)
        conditions.append((query, vals))

    query_parts, values = [], []
    for c in conditions:
        query_parts.append(c[0])
        values.append(c[1])
    if isinstance(values[0], list):
        values = list(chain.from_iterable(values))

    if operator == "AND":
        sql_query = f"({' AND '.join(query_parts)})"
    elif operator == "OR":
        sql_query = f"({' OR '.join(query_parts)})"
    elif operator == "NOT":
        sql_query = f"NOT ({' AND '.join(query_parts)})"
    else:
        msg = f"Unknown logical operator '{operator}'"
        raise FilterError(msg)

    return sql_query, values


def _parse_comparison_condition(condition: Dict[str, Any]) -> tuple[str, List[Any]]:
    field: str = condition["field"]
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "value" not in condition:
        msg = f"'value' key missing in {condition}"
        raise FilterError(msg)
    operator: str = condition["operator"]
    if operator not in COMPARISON_OPERATORS:
        msg = f"Unknown comparison operator '{operator}'. Valid operators are: {list(COMPARISON_OPERATORS.keys())}"
        raise FilterError(msg)

    value: Any = condition["value"]
    if field.startswith("meta."):
        field = _treat_meta_field(field, value)

    field, value = COMPARISON_OPERATORS[operator](field, value)
    return field, [value]


# TODO: determine if we need value for type determination, so leaving parameter here even though unused.


def _treat_meta_field(field: str, value: Any) -> str:  # noqa: ARG001
    """
    Internal method that modifies the field str
    to make the meta JSONB field queryable.

    Examples:
    >>> _treat_meta_field(field="meta.number", value=9)
    "(meta->>'number')::integer"

    >>> _treat_meta_field(field="meta.name", value="my_name")
    "meta->>'name'"
    """

    # use the ->> operator to access keys in the meta JSONB field
    # Well, we would, but that only works with sqlite 3.38 and beyond
    field_name = field.split(".", 1)[-1]
    field = f"json_extract(meta, '$.{field_name}')"

    # TODO: Investigate whether I really need this.
    # meta fields are stored as strings in the JSONB field,
    # so we need to cast them to the correct type
    # type_value = PYTHON_TYPES_TO_PG_TYPES.get(type(value))
    # if isinstance(value, list) and len(value) > 0:
    #    type_value = PYTHON_TYPES_TO_PG_TYPES.get(type(value[0]))

    # if type_value:
    #    field = f"({field})::{type_value}"

    return field


def _equal(field: str, value: Any) -> tuple[str, Any]:
    if value is None:
        # NO_VALUE is a placeholder that will be removed in _convert_filters_to_where_clause_and_params
        return f"{field} IS NULL", NO_VALUE
    return f"{field} = ?", value


def _not_equal(field: str, value: Any) -> tuple[str, Any]:
    # we use IS DISTINCT FROM to correctly handle NULL values
    # (not handled by !=)
    return f"{field} IS NOT ?", value


def _greater_than(field: str, value: Any) -> tuple[str, Any]:
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if type(value) in [list, dict]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)

    return f"{field} > ?", value


def _greater_than_equal(field: str, value: Any) -> tuple[str, Any]:
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if type(value) in [list, dict]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)

    return f"{field} >= ?", value


def _less_than(field: str, value: Any) -> tuple[str, Any]:
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if type(value) in [list, dict]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)

    return f"{field} < ?", value


def _less_than_equal(field: str, value: Any) -> tuple[str, Any]:
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if type(value) in [list, dict]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)

    return f"{field} <= ?", value


def _not_in(field: str, value: Any) -> tuple[str, Any]:
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'not in' comparator in Pinecone"
        raise FilterError(msg)

    # TODO: consider replacing with NOT IN
    value_json = json.dumps(value)
    return f"{field} IS NULL OR {field} NOT IN (select value from json_each(?))", value_json  # noqa: S608


def _in(field: str, value: Any) -> tuple[str, Any]:
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'in' comparator in Pinecone"
        raise FilterError(msg)

    # TODO: consider replacing with IN
    value_json = json.dumps(value)
    return f"{field} IN (select value from json_each(?))", value_json  # noqa: S608


COMPARISON_OPERATORS = {
    "==": _equal,
    "!=": _not_equal,
    ">": _greater_than,
    ">=": _greater_than_equal,
    "<": _less_than,
    "<=": _less_than_equal,
    "in": _in,
    "not in": _not_in,
}
