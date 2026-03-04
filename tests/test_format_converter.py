"""Unit tests for metric format conversion utilities."""

from dctools.utilities.format_converter import (
    convert_format1_to_format2,
    convert_format2_to_format1,
    filter_format2_by_variables,
    group_format2_by_metric,
)


def test_convert_format1_simple_to_format2():
    """Simple format should be converted when metric name is provided."""
    data = {"Surface salinity": [0.78], "50m salinity": 0.36}

    out = convert_format1_to_format2(data, metric_name="rmse")

    assert len(out) == 2
    assert out[0]["Metric"] == "rmse"
    assert {row["Variable"] for row in out} == {"Surface salinity", "50m salinity"}


def test_convert_format1_nested_to_format2():
    """Nested metric dict should be expanded in format2 rows."""
    data = {"rmsd": {"A": 1.0, "B": 2.0}}

    out = convert_format1_to_format2(data)

    assert len(out) == 2
    assert {row["Metric"] for row in out} == {"rmsd"}


def test_convert_format1_simple_requires_metric_name():
    """Simple input without metric_name should return empty result."""
    out = convert_format1_to_format2({"A": [1.0]})
    assert out == []


def test_convert_format2_to_format1_filters_invalid_rows():
    """Invalid rows should be ignored while valid rows are converted."""
    out = convert_format2_to_format1(
        [
            {"Metric": "rmse", "Variable": "A", "Value": "1.5"},
            {"Metric": "rmse", "Variable": "A", "Value": 2.0},
            {"Metric": "rmse", "Variable": "B", "Value": "bad"},
            {"Metric": "rmse", "Value": 0.3},
        ]
    )

    assert out["A"] == [1.5, 2.0]
    assert "B" not in out


def test_group_and_filter_format2_results():
    """Grouping and variable filtering should keep consistent rows."""
    rows = [
        {"Metric": "rmse", "Variable": "A", "Value": 1.0},
        {"Metric": "mae", "Variable": "A", "Value": 0.8},
        {"Metric": "rmse", "Variable": "B", "Value": 2.0},
    ]

    grouped = group_format2_by_metric(rows)
    filtered = filter_format2_by_variables(rows, ["A"])

    assert set(grouped.keys()) == {"rmse", "mae"}
    assert len(grouped["rmse"]) == 2
    assert all(row["Variable"] == "A" for row in filtered)
