//! Profile runtime micro benchmarks.

mod support;

use std::sync::Arc;

use _scdata::profile::{
    ProfileComponent, ProfileConfig, ProfileDefault, ProfileMetricId, ProfilePhase,
    ProfileRegistry, ProfileRuleSet, ProfileScope, ProfileScopeKind, ProfileTimer,
};
use support::{
    bench, bench_profiled, disabled_profile_runtime, metric_value, profile_runtime,
    selected_scope_profile_runtime, stress_mt, BenchConfig,
};

_scdata::scdata_profile_component!(const BENCH_COMPONENT = "bench-profile");
_scdata::scdata_profile_scope!(const HOT_SCOPE = BENCH_COMPONENT, "hot");
_scdata::scdata_profile_scope!(const COLD_SCOPE = BENCH_COMPONENT, "cold");
_scdata::scdata_profile_component!(const LATE_COMPONENT = "bench-late");
_scdata::scdata_profile_scope!(const LATE_SCOPE = LATE_COMPONENT, "late");

const HOT_CALLS: ProfileMetricId = ProfileMetricId::count(HOT_SCOPE, "calls");
const HOT_BYTES: ProfileMetricId = ProfileMetricId::bytes(HOT_SCOPE, "bytes");
const HOT_NS: ProfileMetricId = ProfileMetricId::duration(HOT_SCOPE, "work");
const HOT_GAUGE: ProfileMetricId = ProfileMetricId::gauge(HOT_SCOPE, "gauge");
const COLD_CALLS: ProfileMetricId = ProfileMetricId::count(COLD_SCOPE, "calls");
const LATE_CALLS: ProfileMetricId = ProfileMetricId::count(LATE_SCOPE, "calls");

fn main() {
    let config = BenchConfig::from_env();
    println!("scdata profile runtime benchmarks");
    bench_rule_resolution(config);
    bench_registry_build(config);
    bench_disabled_runtime(config);
    bench_enabled_recording(config);
    bench_selected_scopes(config);
    bench_recorder_api(config);
    bench_late_registry(config);
    bench_macro_measure(config);
    bench_snapshot_reset(config);
    bench_round_lifecycle(config);
    bench_timer_baseline(config);
    bench_threaded_recording(config);
}

fn bench_rule_resolution(config: BenchConfig) {
    let mut component_rules = ProfileRuleSet::only(["bench-profile"]);
    component_rules.disable("unused-*");
    let scope_rules = ProfileRuleSet::from_definitions()
        .with_default(ProfileDefault::Disabled)
        .with_enabled("bench-profile.hot")
        .with_disabled("bench-profile.cold");
    let profile_config = ProfileConfig::enabled("profile-rule-resolution")
        .with_components(component_rules)
        .with_scopes(scope_rules);

    bench(
        config,
        "profile/config_rule_resolution",
        1_000_000,
        None,
        || {
            let hot = profile_config.scope_enabled(HOT_SCOPE, true, true);
            let cold = profile_config.scope_enabled(COLD_SCOPE, true, true);
            let parsed = ProfileRuleSet::parse("bench-profile.* bench-late.*");
            hot as usize ^ ((cold as usize) << 1) ^ parsed.enabled_patterns().len()
        },
    );
}

fn bench_registry_build(config: BenchConfig) {
    bench(
        config,
        "profile/registry_build_and_replace",
        200_000,
        None,
        || {
            let mut registry = ProfileRegistry::new()
                .with_component(
                    ProfileComponent::new(BENCH_COMPONENT).described("first definition"),
                )
                .with_scope(ProfileScope::new(HOT_SCOPE).kind(ProfileScopeKind::Timer));
            registry.add_component(
                ProfileComponent::new(BENCH_COMPONENT).described("replacement definition"),
            );
            registry.add_scope(
                ProfileScope::new(HOT_SCOPE)
                    .kind(ProfileScopeKind::Bytes)
                    .described("replacement scope"),
            );
            registry.components().len() ^ registry.scopes().len()
        },
    );
}

fn bench_disabled_runtime(config: BenchConfig) {
    let runtime = disabled_profile_runtime(bench_registry);
    bench(
        config,
        "profile/disabled_with_recorder",
        1_000_000,
        None,
        || {
            runtime.with_recorder(|ctx| {
                ctx.inc(HOT_CALLS);
                ctx.add(HOT_BYTES, 64);
            });
            1
        },
    );
}

fn bench_enabled_recording(config: BenchConfig) {
    let runtime = profile_runtime("profile-enabled", bench_registry);
    let snapshot = bench_profiled(
        config,
        "profile/enabled_record_counter_timer",
        1_000_000,
        Some(64),
        &runtime,
        || {
            runtime.with_recorder(|ctx| {
                let timer = ctx.timer(HOT_SCOPE);
                ctx.inc(HOT_CALLS);
                ctx.add(HOT_BYTES, 64);
                ctx.record_timer(HOT_NS, timer);
            });
            64
        },
    );
    assert!(metric_value(&snapshot, "bench-profile.hot", "calls") > 0);
}

fn bench_selected_scopes(config: BenchConfig) {
    let runtime =
        selected_scope_profile_runtime("profile-selected", ["bench-profile.hot"], bench_registry);
    let snapshot = bench_profiled(
        config,
        "profile/selected_scope_hot_only",
        1_000_000,
        None,
        &runtime,
        || {
            runtime.with_recorder(|ctx| {
                ctx.inc(HOT_CALLS);
                ctx.inc(COLD_CALLS);
            });
            1
        },
    );
    assert!(metric_value(&snapshot, "bench-profile.hot", "calls") > 0);
    assert_eq!(metric_value(&snapshot, "bench-profile.cold", "calls"), 0);
}

fn bench_recorder_api(config: BenchConfig) {
    let runtime = profile_runtime("profile-recorder-api", bench_registry);
    let snapshot = bench_profiled(
        config,
        "profile/recorder_scope_round_set_add_usize",
        500_000,
        Some(256),
        &runtime,
        || {
            runtime.with_recorder(|ctx| {
                if ctx.is_scope_enabled(HOT_SCOPE) {
                    ctx.inc(HOT_CALLS);
                    ctx.add_usize(HOT_BYTES, 256);
                    ctx.set(HOT_GAUGE, ctx.round());
                }
                if ctx.is_scope_enabled(COLD_SCOPE) {
                    ctx.inc(COLD_CALLS);
                }
            });
            256
        },
    );
    assert!(metric_value(&snapshot, "bench-profile.hot", "gauge") > 0);
}

fn bench_late_registry(config: BenchConfig) {
    let runtime = profile_runtime("profile-late-registry", bench_registry);
    let snapshot = bench_profiled(
        config,
        "profile/ensure_registry_during_round",
        200_000,
        None,
        &runtime,
        || {
            runtime.ensure_registry_lazy(late_registry);
            runtime.with_recorder(|ctx| {
                ctx.inc(LATE_CALLS);
            });
            1
        },
    );
    assert!(metric_value(&snapshot, "bench-late.late", "calls") > 0);
}

fn bench_macro_measure(config: BenchConfig) {
    let runtime = profile_runtime("profile-macro", bench_registry);
    let snapshot = bench_profiled(
        config,
        "profile/macro_measure",
        500_000,
        Some(128),
        &runtime,
        || {
            _scdata::scdata_profile_measure!(
                runtime,
                HOT_SCOPE,
                {
                    let mut sum = 0usize;
                    for i in 0..16 {
                        sum ^= i * 17;
                    }
                    sum
                },
                |ctx, timer, result| {
                    ctx.inc(HOT_CALLS);
                    ctx.add(HOT_BYTES, *result as u64);
                    ctx.record_timer(HOT_NS, timer);
                }
            )
        },
    );
    assert!(metric_value(&snapshot, "bench-profile.hot", "work") > 0);
}

fn bench_snapshot_reset(config: BenchConfig) {
    let runtime = profile_runtime("profile-reset", bench_registry);
    let round = runtime.start();
    bench(
        config,
        "profile/snapshot_and_reset_active_round",
        100_000,
        None,
        || {
            runtime.with_recorder(|ctx| {
                ctx.inc(HOT_CALLS);
                ctx.add(HOT_BYTES, 8);
            });
            let snapshot = runtime.snapshot_and_reset();
            metric_value(&snapshot, "bench-profile.hot", "calls") as usize
        },
    );
    support::maybe_print_profile_snapshot(config, "profile/snapshot_and_reset_final", &round.end());
}

fn bench_round_lifecycle(config: BenchConfig) {
    let runtime = profile_runtime("profile-round-lifecycle", bench_registry);
    bench(
        config,
        "profile/start_end_empty_round",
        200_000,
        None,
        || {
            assert_eq!(runtime.phase(), ProfilePhase::Idle);
            let round = runtime.start();
            assert_eq!(runtime.phase(), ProfilePhase::Recording);
            let snapshot = round.end();
            snapshot.round as usize ^ snapshot.enabled_scope_count()
        },
    );
}

fn bench_timer_baseline(config: BenchConfig) {
    bench(
        config,
        "profile/timer_start_elapsed",
        1_000_000,
        None,
        || {
            let enabled = ProfileTimer::start(true);
            let disabled = ProfileTimer::start(false);
            enabled.is_enabled() as usize ^ disabled.elapsed_ns().unwrap_or(0) as usize
        },
    );
}

fn bench_threaded_recording(config: BenchConfig) {
    let runtime = Arc::new(profile_runtime("profile-threaded", bench_registry));
    let round = runtime.start();
    for threads in [2, 4, 8] {
        let runtime = Arc::clone(&runtime);
        stress_mt(
            config,
            "profile/threaded_record",
            threads,
            1_000_000,
            Some(32),
            move |_| {
                runtime.with_recorder(|ctx| {
                    ctx.inc(HOT_CALLS);
                    ctx.add(HOT_BYTES, 32);
                });
                32
            },
        );
    }
    support::maybe_print_profile_snapshot(config, "profile/threaded_record", &round.end());
}

fn bench_registry() -> ProfileRegistry {
    ProfileRegistry::new()
        .with_component(ProfileComponent::new(BENCH_COMPONENT).described("bench profile runtime"))
        .with_scope(
            ProfileScope::new(HOT_SCOPE)
                .kind(ProfileScopeKind::Timer)
                .described("hot profile path"),
        )
        .with_scope(
            ProfileScope::new(COLD_SCOPE)
                .kind(ProfileScopeKind::Counter)
                .described("scope disabled in selected-scope runs"),
        )
}

fn late_registry() -> ProfileRegistry {
    ProfileRegistry::new()
        .with_component(ProfileComponent::new(LATE_COMPONENT).described("late-bound component"))
        .with_scope(
            ProfileScope::new(LATE_SCOPE)
                .kind(ProfileScopeKind::Event)
                .described("late-bound scope"),
        )
}
