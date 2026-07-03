//! Process-level cached environment configuration.
//!
//! These getters are for env knobs whose values are intended to stay fixed for
//! a process run. Each registered env is parsed once, then served from a
//! `OnceLock` on subsequent reads.

fn bool_default_off(name: &'static str) -> bool {
    std::env::var(name)
        .map(|value| is_truthy(value.as_str()))
        .unwrap_or(false)
}

fn bool_default_on(name: &'static str) -> bool {
    std::env::var(name)
        .map(|value| !is_falsy(value.as_str()))
        .unwrap_or(true)
}

fn usize_env(name: &'static str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
}

fn is_truthy(value: &str) -> bool {
    matches!(value, "1" | "true" | "TRUE" | "yes" | "on")
}

fn is_falsy(value: &str) -> bool {
    matches!(value, "0" | "false" | "FALSE" | "no" | "off")
}

macro_rules! register_bool_env {
    ($(#[$meta:meta])* $vis:vis fn $fn_name:ident() = $name:literal, default_off $(,)?) => {
        $(#[$meta])*
        $vis fn $fn_name() -> bool {
            static VALUE: ::std::sync::OnceLock<bool> = ::std::sync::OnceLock::new();
            *VALUE.get_or_init(|| $crate::env::bool_default_off($name))
        }
    };
    ($(#[$meta:meta])* $vis:vis fn $fn_name:ident() = $name:literal, default_on $(,)?) => {
        $(#[$meta])*
        $vis fn $fn_name() -> bool {
            static VALUE: ::std::sync::OnceLock<bool> = ::std::sync::OnceLock::new();
            *VALUE.get_or_init(|| $crate::env::bool_default_on($name))
        }
    };
}

macro_rules! register_usize_env {
    ($(#[$meta:meta])* $vis:vis fn $fn_name:ident() = $name:literal, any $(,)?) => {
        $(#[$meta])*
        $vis fn $fn_name() -> Option<usize> {
            static VALUE: ::std::sync::OnceLock<Option<usize>> = ::std::sync::OnceLock::new();
            *VALUE.get_or_init(|| $crate::env::usize_env($name))
        }
    };
    ($(#[$meta:meta])* $vis:vis fn $fn_name:ident() = $name:literal, gt($min:expr) $(,)?) => {
        $(#[$meta])*
        $vis fn $fn_name() -> Option<usize> {
            static VALUE: ::std::sync::OnceLock<Option<usize>> = ::std::sync::OnceLock::new();
            *VALUE.get_or_init(|| $crate::env::usize_env($name).filter(|&value| value > $min))
        }
    };
}

pub(crate) mod fastpath {
    register_bool_env!(
        /// Enables grouping native scheduled commands across command boundaries.
        pub(crate) fn native_command_grouping_enabled()
            = "SCDATA_NATIVE_COMMAND_GROUPING",
            default_off
    );

    register_bool_env!(
        /// Allows single-item commands to enter native command grouping.
        pub(crate) fn native_group_single_item_commands_enabled()
            = "SCDATA_NATIVE_GROUP_SINGLE_ITEM_COMMANDS",
            default_off
    );

    register_bool_env!(
        /// Enables in-flight native payload read de-duplication.
        pub(crate) fn native_in_flight_payload_reads_enabled()
            = "SCDATA_NATIVE_INFLIGHT_PAYLOAD_READS",
            default_on
    );

    register_bool_env!(
        /// Enables targeted native caches for selected sparse data commands.
        pub(crate) fn native_targeted_selected_sparse_cache_enabled()
            = "SCDATA_NATIVE_TARGETED_SELECTED_SPARSE_CACHE",
            default_off
    );

    register_bool_env!(
        /// Enables fused native load and scatter for selected sparse data.
        pub(crate) fn native_fused_scatter_enabled()
            = "SCDATA_NATIVE_FUSED_SCATTER",
            default_off
    );

    register_bool_env!(
        /// Enables selected-plan scatter after read-all projected sparse loads.
        pub(crate) fn read_all_selected_scatter_enabled()
            = "SCDATA_READALL_SELECTED_SCATTER",
            default_off
    );

    register_bool_env!(
        /// Enables selected sparse index preplanning.
        pub(crate) fn preplan_selected_sparse_enabled()
            = "SCDATA_PREPLAN_SELECTED_SPARSE",
            default_on
    );

    register_bool_env!(
        /// Defers preplanned selected sparse data to response-time native reads.
        pub(crate) fn preplan_selected_sparse_defer_data_enabled()
            = "SCDATA_PREPLAN_SELECTED_SPARSE_DEFER_DATA",
            default_off
    );

    register_usize_env!(
        /// Overrides the native item batch size before caller-side clamping.
        pub(crate) fn native_item_batch_size()
            = "SCDATA_NATIVE_ITEM_BATCH_SIZE",
            gt(0)
    );

    register_usize_env!(
        /// Overrides cross-command grouping max items before caller-side clamping.
        pub(crate) fn native_command_group_max_items()
            = "SCDATA_NATIVE_COMMAND_GROUP_MAX_ITEMS",
            gt(1)
    );

    register_usize_env!(
        /// Overrides targeted native payload cache bytes. Zero is meaningful.
        pub(crate) fn native_targeted_payload_cache_bytes()
            = "SCDATA_NATIVE_TARGETED_PAYLOAD_CACHE_BYTES",
            any
    );

    register_usize_env!(
        /// Overrides targeted native decoded cache bytes. Zero is meaningful.
        pub(crate) fn native_targeted_decoded_cache_bytes()
            = "SCDATA_NATIVE_TARGETED_DECODED_CACHE_BYTES",
            any
    );
}

#[cfg(test)]
mod tests {
    use super::{is_falsy, is_truthy};

    #[test]
    fn truthy_values_match_fastpath_semantics() {
        for value in ["1", "true", "TRUE", "yes", "on"] {
            assert!(is_truthy(value));
        }
        for value in ["0", "false", "FALSE", "no", "off", "yes please", "True"] {
            assert!(!is_truthy(value));
        }
    }

    #[test]
    fn falsy_values_match_fastpath_semantics() {
        for value in ["0", "false", "FALSE", "no", "off"] {
            assert!(is_falsy(value));
        }
        for value in ["1", "true", "TRUE", "yes", "on", "False"] {
            assert!(!is_falsy(value));
        }
    }
}
