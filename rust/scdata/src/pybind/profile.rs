use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::profile::ProfileSnapshot;

pub(crate) fn snapshot_to_py(py: Python<'_>, snapshot: ProfileSnapshot) -> PyResult<PyObject> {
    let out = PyDict::new(py);
    out.set_item("label", &snapshot.label)?;
    out.set_item("round", snapshot.round)?;
    out.set_item("elapsed_ns", snapshot.elapsed_ns)?;
    out.set_item("elapsed_ms", snapshot.elapsed_ms())?;
    out.set_item("global_enabled", snapshot.global_enabled)?;
    out.set_item("components", components_to_py(py, &snapshot)?)?;
    out.set_item("scopes", scopes_to_py(py, &snapshot)?)?;
    out.set_item("metrics", metrics_to_py(py, &snapshot)?)?;
    Ok(out.into_any().unbind())
}

fn components_to_py(py: Python<'_>, snapshot: &ProfileSnapshot) -> PyResult<PyObject> {
    let items = PyList::empty(py);
    for component in &snapshot.components {
        let item = PyDict::new(py);
        item.set_item("id", component.id.as_str())?;
        item.set_item("enabled", component.enabled)?;
        item.set_item("default_enabled", component.default_enabled)?;
        item.set_item("description", &component.description)?;
        items.append(item)?;
    }
    Ok(items.into_any().unbind())
}

fn scopes_to_py(py: Python<'_>, snapshot: &ProfileSnapshot) -> PyResult<PyObject> {
    let items = PyList::empty(py);
    for scope in &snapshot.scopes {
        let item = PyDict::new(py);
        item.set_item("id", scope.id.full_name())?;
        item.set_item("component", scope.id.component().as_str())?;
        item.set_item("name", scope.id.name())?;
        item.set_item("enabled", scope.enabled)?;
        item.set_item("default_enabled", scope.default_enabled)?;
        item.set_item("kind", scope.kind.as_str())?;
        item.set_item("description", &scope.description)?;
        items.append(item)?;
    }
    Ok(items.into_any().unbind())
}

fn metrics_to_py(py: Python<'_>, snapshot: &ProfileSnapshot) -> PyResult<PyObject> {
    let items = PyList::empty(py);
    for metric in &snapshot.metrics {
        let item = PyDict::new(py);
        item.set_item("scope", metric.id.scope.full_name())?;
        item.set_item("component", metric.id.scope.component().as_str())?;
        item.set_item("scope_name", metric.id.scope.name())?;
        item.set_item("name", metric.id.name)?;
        item.set_item("kind", metric.kind().as_str())?;
        item.set_item("value", metric.value())?;
        item.set_item("ms", metric.as_ms())?;
        item.set_item("mib", metric.as_mib())?;
        items.append(item)?;
    }
    Ok(items.into_any().unbind())
}
