use std::sync::atomic::{AtomicBool, Ordering};

use super::*;

const COMPONENT: ProfileComponentId = ProfileComponentId::new("disabled");
const SCOPE: ProfileScopeId = ProfileScopeId::new(COMPONENT, "scope");
const _METRIC: ProfileMetricId = ProfileMetricId::count(SCOPE, "calls");

#[test]
fn lazy_registry_is_not_evaluated_when_feature_is_disabled() {
    let called = AtomicBool::new(false);
    let runtime = ProfileRuntime::from_env_lazy(|| {
        called.store(true, Ordering::SeqCst);
        ProfileRegistry::new().with_scope(ProfileScope::new(SCOPE))
    });

    assert!(!called.load(Ordering::SeqCst));
    assert!(!runtime.is_global_enabled());
}

#[test]
fn profile_record_macro_drops_record_block_when_feature_is_disabled() {
    let _runtime = ProfileRuntime::disabled();
    let called = AtomicBool::new(false);

    crate::scdata_profile_record!(_runtime, |ctx| {
        called.store(true, Ordering::SeqCst);
        ctx.inc(_METRIC);
        let _ = profile_only_symbol_that_must_not_be_resolved;
    });

    assert!(!called.load(Ordering::SeqCst));
}

#[test]
fn profile_measure_macro_keeps_work_and_drops_record_block_when_feature_is_disabled() {
    let _runtime = ProfileRuntime::disabled();
    let called = AtomicBool::new(false);

    let value =
        crate::scdata_profile_measure!(_runtime, SCOPE, { 7usize }, |ctx, timer, result| {
            called.store(true, Ordering::SeqCst);
            assert!(timer.is_enabled());
            ctx.add_usize(_METRIC, *result);
            let _ = profile_only_symbol_that_must_not_be_resolved;
        });

    assert_eq!(value, 7);
    assert!(!called.load(Ordering::SeqCst));
}
