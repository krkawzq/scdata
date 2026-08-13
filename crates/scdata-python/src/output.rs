//! Build a Rust output specification from normalized Python scalars.

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use sc_load::{Fill, FloatCastPolicy, OutputDType, OutputSpec, OverflowPolicy};

use crate::error::{invalid_input as invalid_argument, ResultExt};

pub(crate) fn output_spec_from_dict(values: &Bound<'_, PyDict>) -> PyResult<OutputSpec> {
    let dtype = required(values, "dtype")?
        .extract::<String>()?
        .parse::<OutputDType>()
        .map_sc()?;
    let fill = extract_fill(dtype, &required(values, "fill")?)?;
    let overflow_name = required(values, "overflow")?.extract::<String>()?;
    let overflow_value = values.get_item("overflow_value")?;
    let overflow = match overflow_name.as_str() {
        "error" => OverflowPolicy::Error,
        "use_fill" => OverflowPolicy::UseFill,
        "use_value" => OverflowPolicy::UseValue(extract_fill(
            dtype,
            &overflow_value
                .filter(|value| !value.is_none())
                .ok_or_else(|| {
                    invalid_argument("overflow_value is required when overflow='use_value'")
                })?,
        )?),
        "unchecked" => OverflowPolicy::Unchecked,
        other => {
            return Err(invalid_argument(format!(
                "unknown overflow policy `{other}`; expected 'error', 'use_fill', 'use_value', or 'unchecked'"
            )))
        }
    };
    let float_cast = if required(values, "allow_float_rounding")?.extract()? {
        FloatCastPolicy::AllowRounding
    } else {
        FloatCastPolicy::ExactOnly
    };
    let output = OutputSpec::new(required(values, "n_cols")?.extract()?, dtype, fill)
        .map_sc()?
        .overflow(overflow)
        .map_sc()?;
    Ok(output.float_cast(float_cast))
}

fn required<'py>(values: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    values
        .get_item(key)?
        .ok_or_else(|| invalid_argument(format!("missing normalized output field `{key}`")))
}

fn extract_fill(dtype: OutputDType, value: &Bound<'_, PyAny>) -> PyResult<Fill> {
    match dtype {
        OutputDType::I16 => value.extract().map(Fill::I16),
        OutputDType::I32 => value.extract().map(Fill::I32),
        OutputDType::I64 => value.extract().map(Fill::I64),
        OutputDType::U16 => value.extract().map(Fill::U16),
        OutputDType::U32 => value.extract().map(Fill::U32),
        OutputDType::U64 => value.extract().map(Fill::U64),
        OutputDType::F32 => value.extract().map(Fill::F32),
        OutputDType::F64 => value.extract().map(Fill::F64),
    }
}
