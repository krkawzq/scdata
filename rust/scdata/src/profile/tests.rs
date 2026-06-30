use std::sync::atomic::{AtomicBool, Ordering};

use super::*;

const CPU: ProfileComponentId = ProfileComponentId::new("cpu");
const IO: ProfileComponentId = ProfileComponentId::new("io");
const CPU_HOT_LOOP: ProfileScopeId = ProfileScopeId::new(CPU, "hot-loop");
const CPU_COLD_PATH: ProfileScopeId = ProfileScopeId::new(CPU, "cold-path");
const IO_READ: ProfileScopeId = ProfileScopeId::new(IO, "read");

const HOT_LOOP_COUNT: ProfileMetricId = ProfileMetricId::count(CPU_HOT_LOOP, "iterations");
const HOT_LOOP_NS: ProfileMetricId = ProfileMetricId::duration(CPU_HOT_LOOP, "elapsed");
const IO_BYTES: ProfileMetricId = ProfileMetricId::bytes(IO_READ, "bytes");

crate::scdata_profile_component!(const MACRO_COMPONENT = "macro");
crate::scdata_profile_scope!(const MACRO_SCOPE = MACRO_COMPONENT, "fast-path");

fn registry() -> ProfileRegistry {
    ProfileRegistry::new()
        .with_component(ProfileComponent::new(CPU).described("CPU work"))
        .with_component(ProfileComponent::new(IO).described("IO work"))
        .with_scope(
            ProfileScope::new(CPU_HOT_LOOP)
                .kind(ProfileScopeKind::Counter)
                .described("hot loop iterations"),
        )
        .with_scope(ProfileScope::new(CPU_COLD_PATH).disabled_by_default())
        .with_scope(ProfileScope::new(IO_READ).kind(ProfileScopeKind::Bytes))
}

#[test]
fn lazy_registry_is_evaluated_when_profile_is_enabled() {
    let called = AtomicBool::new(false);
    let profiler = ProfileRuntime::new_lazy(ProfileConfig::enabled("lazy"), || {
        called.store(true, Ordering::SeqCst);
        registry()
    });

    assert!(called.load(Ordering::SeqCst));
    let round = profiler.start();
    assert!(profiler.with_recorder(|ctx| ctx.is_scope_enabled(CPU_HOT_LOOP)) == Some(true));
    round.end();
}

#[test]
fn recorder_api_records_only_during_active_rounds() {
    let profiler = ProfileRuntime::new_lazy(ProfileConfig::enabled("recorder"), registry);
    assert_eq!(
        profiler.with_recorder(|ctx| {
            ctx.inc(HOT_LOOP_COUNT);
            1
        }),
        None
    );

    let round = profiler.start();
    let value = profiler.with_recorder(|ctx| {
        assert_eq!(ctx.round(), round.round());
        ctx.inc(HOT_LOOP_COUNT);
        ctx.add(HOT_LOOP_COUNT, 2);
        ctx.set(IO_BYTES, 64);
        42
    });

    assert_eq!(value, Some(42));
    let snapshot = round.end();
    assert_eq!(snapshot.metric_value(HOT_LOOP_COUNT), Some(3));
    assert_eq!(snapshot.metric_value(IO_BYTES), Some(64));
}

#[test]
fn snapshot_and_reset_keeps_the_round_recording() {
    let profiler = ProfileRuntime::new_lazy(ProfileConfig::enabled("snapshot-reset"), registry);
    let round = profiler.start();

    crate::scdata_profile_record!(profiler, |ctx| {
        ctx.add(HOT_LOOP_COUNT, 7);
    });
    let reset = profiler.snapshot_and_reset();
    assert!(profiler.is_recording());
    assert_eq!(reset.metric_value(HOT_LOOP_COUNT), Some(7));

    crate::scdata_profile_record!(profiler, |ctx| {
        ctx.add(HOT_LOOP_COUNT, 2);
    });
    assert_eq!(round.end().metric_value(HOT_LOOP_COUNT), Some(2));
}

#[test]
fn record_macro_uses_recorder_without_exposing_handles() {
    let profiler = ProfileRuntime::new_lazy(ProfileConfig::enabled("macro-record"), registry);
    let round = profiler.start();

    crate::scdata_profile_record!(profiler, |ctx| {
        ctx.add(HOT_LOOP_COUNT, 4);
    });

    let snapshot = round.end();
    assert_eq!(snapshot.metric_value(HOT_LOOP_COUNT), Some(4));
}

#[test]
fn measure_macro_runs_body_and_records_afterwards() {
    let profiler = ProfileRuntime::new_lazy(ProfileConfig::enabled("measure"), registry);
    let round = profiler.start();

    let value =
        crate::scdata_profile_measure!(profiler, CPU_HOT_LOOP, { 5usize }, |ctx, timer, result| {
            assert!(timer.is_enabled());
            ctx.add_usize(HOT_LOOP_COUNT, *result);
            ctx.record_timer(HOT_LOOP_NS, timer);
        });

    assert_eq!(value, 5);
    let snapshot = round.end();
    assert_eq!(snapshot.metric_value(HOT_LOOP_COUNT), Some(5));
    assert!(snapshot.metric_value(HOT_LOOP_NS).is_some_and(|ns| ns > 0));
}

#[test]
fn disabled_timer_recording_does_not_allocate_metric() {
    let profiler = ProfileRuntime::new_lazy(ProfileConfig::enabled("disabled-timer"), registry);
    let round = profiler.start();

    crate::scdata_profile_record!(profiler, |ctx| {
        ctx.record_timer(HOT_LOOP_NS, ProfileTimer::disabled());
    });

    assert_eq!(round.end().metric_value(HOT_LOOP_NS), None);
}

#[test]
fn measure_macro_preserves_question_mark_in_body() -> Result<(), &'static str> {
    let profiler = ProfileRuntime::new_lazy(ProfileConfig::enabled("measure-question"), registry);
    let round = profiler.start();

    let value = crate::scdata_profile_measure!(
        profiler,
        CPU_HOT_LOOP,
        { Ok::<usize, &'static str>(11)? },
        |ctx, _timer, result| {
            ctx.add_usize(HOT_LOOP_COUNT, *result);
        }
    );

    assert_eq!(value, 11);
    assert_eq!(round.end().metric_value(HOT_LOOP_COUNT), Some(11));
    Ok(())
}

#[test]
fn disabled_scope_does_not_allocate_metric() {
    let profiler = ProfileRuntime::new_lazy(ProfileConfig::enabled("defaults"), registry);
    let round = profiler.start();

    crate::scdata_profile_record!(profiler, |ctx| {
        assert!(!ctx.is_scope_enabled(CPU_COLD_PATH));
        ctx.add(ProfileMetricId::count(CPU_COLD_PATH, "calls"), 9);
    });

    let snapshot = round.end();
    assert_eq!(
        snapshot.metric_value(ProfileMetricId::count(CPU_COLD_PATH, "calls")),
        None
    );
}

#[test]
fn measure_macro_skips_record_block_when_scope_is_disabled() {
    let profiler = ProfileRuntime::new_lazy(ProfileConfig::enabled("measure-disabled"), registry);
    let called = AtomicBool::new(false);
    let cold_calls = ProfileMetricId::count(CPU_COLD_PATH, "calls");
    let round = profiler.start();

    let value = crate::scdata_profile_measure!(
        profiler,
        CPU_COLD_PATH,
        { 3usize },
        |ctx, _timer, result| {
            called.store(true, Ordering::SeqCst);
            ctx.add_usize(cold_calls, *result);
        }
    );

    let snapshot = round.end();
    assert_eq!(value, 3);
    assert!(!called.load(Ordering::SeqCst));
    assert_eq!(snapshot.metric_value(cold_calls), None);
}

#[test]
fn global_disabled_runtime_does_not_run_record_closures() {
    let profiler = ProfileRuntime::new_lazy(ProfileConfig::disabled(), registry);
    let called = AtomicBool::new(false);
    let round = profiler.start();

    assert_eq!(
        profiler.with_recorder(|ctx| {
            called.store(true, Ordering::SeqCst);
            ctx.inc(HOT_LOOP_COUNT);
        }),
        None
    );
    crate::scdata_profile_record!(profiler, |ctx| {
        called.store(true, Ordering::SeqCst);
        ctx.inc(HOT_LOOP_COUNT);
    });

    let snapshot = round.end();
    assert!(!called.load(Ordering::SeqCst));
    assert_eq!(snapshot.metric_value(HOT_LOOP_COUNT), None);
}

#[test]
fn stale_recorder_cannot_write_into_later_rounds() {
    let profiler = ProfileRuntime::new_lazy(ProfileConfig::enabled("stale-recorder"), registry);
    let round = profiler.start();
    let recorder = profiler.with_recorder(|ctx| ctx.clone()).unwrap();
    round.end();

    let round = profiler.start();
    recorder.inc(HOT_LOOP_COUNT);

    let snapshot = round.end();
    assert_eq!(snapshot.metric_value(HOT_LOOP_COUNT), None);
}

#[test]
fn metrics_do_not_leak_into_later_rounds() {
    let profiler = ProfileRuntime::new_lazy(ProfileConfig::enabled("round-metrics"), registry);
    let round = profiler.start();
    crate::scdata_profile_record!(profiler, |ctx| {
        ctx.inc(HOT_LOOP_COUNT);
    });
    assert_eq!(round.end().metric_value(HOT_LOOP_COUNT), Some(1));

    let round = profiler.start();
    let snapshot = round.end();
    assert_eq!(snapshot.metric_value(HOT_LOOP_COUNT), None);
    assert!(snapshot.metrics.is_empty());
}

#[test]
fn idle_snapshot_is_empty_after_round_end() {
    let profiler = ProfileRuntime::new_lazy(ProfileConfig::enabled("idle-snapshot"), registry);
    let round = profiler.start();
    crate::scdata_profile_record!(profiler, |ctx| {
        ctx.inc(HOT_LOOP_COUNT);
    });
    round.end();

    let snapshot = profiler.snapshot();
    assert_eq!(snapshot.round, 0);
    assert!(snapshot.metrics.is_empty());
}

#[test]
fn measure_macro_keeps_recording_bound_to_the_original_round() {
    let profiler = ProfileRuntime::new_lazy(ProfileConfig::enabled("measure-stale"), registry);
    let _round = profiler.start();

    let value = crate::scdata_profile_measure!(
        profiler,
        CPU_HOT_LOOP,
        {
            profiler.end();
            let second = profiler.start();
            std::mem::forget(second);
            9usize
        },
        |ctx, timer, result| {
            assert_eq!(*result, 9);
            assert!(timer.is_enabled());
            ctx.inc(HOT_LOOP_COUNT);
            ctx.record_timer(HOT_LOOP_NS, timer);
        }
    );

    assert_eq!(value, 9);
    let snapshot = profiler.end();
    assert_eq!(snapshot.metric_value(HOT_LOOP_COUNT), None);
    assert_eq!(snapshot.metric_value(HOT_LOOP_NS), None);
}

#[test]
fn config_changes_apply_to_the_next_round() {
    let profiler = ProfileRuntime::new_lazy(
        ProfileConfig::enabled("rounds").with_scopes(ProfileRuleSet::none()),
        registry,
    );

    let round = profiler.start();
    crate::scdata_profile_record!(profiler, |ctx| {
        assert!(!ctx.is_scope_enabled(CPU_HOT_LOOP));
        ctx.inc(HOT_LOOP_COUNT);
    });
    assert_eq!(round.end().metric_value(HOT_LOOP_COUNT), None);

    profiler.update_config(
        ProfileConfig::enabled("rounds").with_scopes(ProfileRuleSet::only(["cpu.hot-loop"])),
    );
    let round = profiler.start();
    crate::scdata_profile_record!(profiler, |ctx| {
        assert!(ctx.is_scope_enabled(CPU_HOT_LOOP));
        ctx.inc(HOT_LOOP_COUNT);
    });
    assert_eq!(round.end().metric_value(HOT_LOOP_COUNT), Some(1));
}

#[test]
fn ensure_registry_lazy_can_add_scopes_during_a_round() {
    let late_component = ProfileComponentId::new("late");
    let late_scope = ProfileScopeId::new(late_component, "work");
    let late_metric = ProfileMetricId::count(late_scope, "calls");

    let profiler = ProfileRuntime::new_lazy(ProfileConfig::enabled("late"), ProfileRegistry::new);
    let round = profiler.start();

    profiler.ensure_registry_lazy(|| {
        ProfileRegistry::new()
            .with_component(ProfileComponent::new(late_component))
            .with_scope(ProfileScope::new(late_scope))
    });
    crate::scdata_profile_record!(profiler, |ctx| {
        assert!(ctx.is_scope_enabled(late_scope));
        ctx.inc(late_metric);
    });

    let snapshot = round.end();
    assert_eq!(snapshot.metric_value(late_metric), Some(1));
}

#[test]
fn ensure_registry_lazy_adds_component_flags_for_scope_only_registry() {
    let late_component = ProfileComponentId::new("late-scope-only");
    let late_scope = ProfileScopeId::new(late_component, "work");
    let late_metric = ProfileMetricId::count(late_scope, "calls");

    let profiler = ProfileRuntime::new_lazy(ProfileConfig::enabled("late-scope-only"), || {
        ProfileRegistry::new()
    });
    let round = profiler.start();

    profiler
        .ensure_registry_lazy(|| ProfileRegistry::new().with_scope(ProfileScope::new(late_scope)));
    crate::scdata_profile_record!(profiler, |ctx| {
        assert!(ctx.is_scope_enabled(late_scope));
        ctx.inc(late_metric);
    });

    let snapshot = round.end();
    let component = snapshot
        .components
        .iter()
        .find(|component| component.id == late_component)
        .expect("late component snapshot");
    assert!(component.enabled);
    assert_eq!(snapshot.metric_value(late_metric), Some(1));
}

#[test]
fn registry_definitions_are_deduplicated_by_id() {
    let component = ProfileComponentId::new("dedupe");
    let scope = ProfileScopeId::new(component, "scope");

    let registry = ProfileRegistry::new()
        .with_component(ProfileComponent::new(component).disabled_by_default())
        .with_component(ProfileComponent::new(component).enabled_by_default())
        .with_scope(ProfileScope::new(scope).disabled_by_default())
        .with_scope(ProfileScope::new(scope).enabled_by_default());

    assert_eq!(registry.components().len(), 1);
    assert_eq!(registry.scopes().len(), 1);
    assert!(registry.components()[0].default_enabled);
    assert!(registry.scopes()[0].default_enabled);
}

#[test]
fn macro_declared_ids_can_build_a_registry() {
    let registry = crate::scdata_profile_registry!(
        components: [ProfileComponent::new(MACRO_COMPONENT)],
        scopes: [ProfileScope::new(MACRO_SCOPE)],
    );
    let profiler = ProfileRuntime::new_lazy(ProfileConfig::enabled("macro"), || registry);
    let round = profiler.start();

    let enabled = profiler.with_recorder(|ctx| ctx.is_scope_enabled(MACRO_SCOPE));
    assert_eq!(enabled, Some(true));
    round.end();
}

#[test]
fn guard_drop_ends_round_on_early_return() {
    let profiler = ProfileRuntime::new_lazy(ProfileConfig::enabled("guard"), registry);
    {
        let _round = profiler.start();
        assert!(profiler.is_recording());
    }
    assert_eq!(profiler.phase(), ProfilePhase::Idle);
}

#[test]
#[should_panic(expected = "already recording")]
fn start_reentry_panics() {
    let profiler = ProfileRuntime::new_lazy(ProfileConfig::enabled("lock"), registry);
    let _round = profiler.start();
    profiler.start();
}

#[test]
#[should_panic(expected = "not recording")]
fn end_panics_when_idle() {
    let profiler = ProfileRuntime::new_lazy(ProfileConfig::enabled("lock"), registry);
    profiler.end();
}
