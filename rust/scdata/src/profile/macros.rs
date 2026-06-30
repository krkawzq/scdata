#[macro_export]
macro_rules! scdata_profile_component {
    ($vis:vis const $name:ident = $component:literal $(,)?) => {
        $vis const $name: $crate::profile::ProfileComponentId =
            $crate::profile::ProfileComponentId::new($component);
    };
}

#[macro_export]
macro_rules! scdata_profile_scope {
    ($vis:vis const $name:ident = $component:expr, $scope:literal $(,)?) => {
        $vis const $name: $crate::profile::ProfileScopeId =
            $crate::profile::ProfileScopeId::new($component, $scope);
    };
}

#[macro_export]
macro_rules! scdata_profile_registry {
    (
        components: [$($component:expr),* $(,)?],
        scopes: [$($scope:expr),* $(,)?] $(,)?
    ) => {{
        let registry = $crate::profile::ProfileRegistry::new();
        $(let registry = registry.with_component($component);)*
        $(let registry = registry.with_scope($scope);)*
        registry
    }};
}

#[cfg(feature = "profile")]
/// Runs a profiling-only block when the runtime is currently recording.
///
/// The block receives a `ProfileRecorder`. When the `profile` feature is
/// disabled, this macro expands to `()` and does not type-check the block.
#[macro_export]
macro_rules! scdata_profile_record {
    ($profile:expr, |$ctx:ident| $body:block $(,)?) => {{
        let __scdata_profile_runtime = &$profile;
        let _ = __scdata_profile_runtime.with_recorder(|$ctx| $body);
    }};
}

#[cfg(not(feature = "profile"))]
/// Runs a profiling-only block when the runtime is currently recording.
///
/// The block receives a `ProfileRecorder`. When the `profile` feature is
/// disabled, this macro expands to `()` and does not type-check the block.
#[macro_export]
macro_rules! scdata_profile_record {
    ($profile:expr, |$ctx:ident| $body:block $(,)?) => {{
        ()
    }};
}

#[cfg(feature = "profile")]
/// Measures a body and runs a profiling-only recording block afterwards.
///
/// The body always runs. The recording block runs only when the measured scope
/// is enabled, and receives a `ProfileRecorder`, the timer binding, and a
/// reference to the body result. When the `profile` feature is disabled, only
/// the body is emitted.
#[macro_export]
macro_rules! scdata_profile_measure {
    ($profile:expr, $scope:expr, $body:block, |$ctx:ident, $timer:ident, $result:ident| $record:block $(,)?) => {{
        let __scdata_profile_runtime = &$profile;
        let __scdata_profile_scope = $scope;
        match __scdata_profile_runtime.with_recorder(|__scdata_profile_ctx| __scdata_profile_ctx) {
            Some($ctx) => {
                let $timer = $ctx.timer(__scdata_profile_scope);
                if $timer.is_enabled() {
                    let __scdata_profile_result = $body;
                    {
                        let $result = &__scdata_profile_result;
                        $record
                    }
                    __scdata_profile_result
                } else {
                    $body
                }
            }
            _ => $body,
        }
    }};
}

#[cfg(not(feature = "profile"))]
/// Measures a body and runs a profiling-only recording block afterwards.
///
/// The body always runs. The recording block runs only when the measured scope
/// is enabled, and receives a `ProfileRecorder`, the timer binding, and a
/// reference to the body result. When the `profile` feature is disabled, only
/// the body is emitted.
#[macro_export]
macro_rules! scdata_profile_measure {
    ($profile:expr, $scope:expr, $body:block, |$ctx:ident, $timer:ident, $result:ident| $record:block $(,)?) => {{
        $body
    }};
}
