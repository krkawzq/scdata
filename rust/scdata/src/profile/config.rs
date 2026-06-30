use std::fmt;

use super::ids::{split_patterns, ProfileComponentId, ProfilePattern, ProfileScopeId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProfileDefault {
    Enabled,
    Disabled,
    FromDefinition,
}

impl ProfileDefault {
    pub const fn resolve(self, definition_default: bool) -> bool {
        match self {
            Self::Enabled => true,
            Self::Disabled => false,
            Self::FromDefinition => definition_default,
        }
    }
}

impl fmt::Display for ProfileDefault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Enabled => f.write_str("enabled"),
            Self::Disabled => f.write_str("disabled"),
            Self::FromDefinition => f.write_str("definition"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ProfileRuleSet {
    default: ProfileDefault,
    enabled: Vec<ProfilePattern>,
    disabled: Vec<ProfilePattern>,
}

impl Default for ProfileRuleSet {
    fn default() -> Self {
        Self::from_definitions()
    }
}

impl ProfileRuleSet {
    pub fn from_definitions() -> Self {
        Self {
            default: ProfileDefault::FromDefinition,
            enabled: Vec::new(),
            disabled: Vec::new(),
        }
    }

    pub fn all() -> Self {
        Self {
            default: ProfileDefault::Enabled,
            enabled: Vec::new(),
            disabled: Vec::new(),
        }
    }

    pub fn none() -> Self {
        Self {
            default: ProfileDefault::Disabled,
            enabled: Vec::new(),
            disabled: Vec::new(),
        }
    }

    pub fn only(patterns: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let mut rules = Self::none();
        for pattern in patterns {
            rules.enable(pattern);
        }
        rules
    }

    pub fn with_default(mut self, default: ProfileDefault) -> Self {
        self.default = default;
        self
    }

    pub fn default_mode(&self) -> ProfileDefault {
        self.default
    }

    pub fn enable(&mut self, pattern: impl Into<String>) {
        self.enabled.push(ProfilePattern::new(pattern));
    }

    pub fn disable(&mut self, pattern: impl Into<String>) {
        self.disabled.push(ProfilePattern::new(pattern));
    }

    pub fn with_enabled(mut self, pattern: impl Into<String>) -> Self {
        self.enable(pattern);
        self
    }

    pub fn with_disabled(mut self, pattern: impl Into<String>) -> Self {
        self.disable(pattern);
        self
    }

    pub fn enabled_patterns(&self) -> &[ProfilePattern] {
        &self.enabled
    }

    pub fn disabled_patterns(&self) -> &[ProfilePattern] {
        &self.disabled
    }

    pub fn resolve_component(
        &self,
        component: ProfileComponentId,
        definition_default: bool,
    ) -> bool {
        if self
            .disabled
            .iter()
            .any(|pattern| pattern.matches_component(component))
        {
            return false;
        }
        if self
            .enabled
            .iter()
            .any(|pattern| pattern.matches_component(component))
        {
            return true;
        }
        self.default.resolve(definition_default)
    }

    pub fn resolve_scope(&self, scope: ProfileScopeId, definition_default: bool) -> bool {
        if self
            .disabled
            .iter()
            .any(|pattern| pattern.matches_scope(scope))
        {
            return false;
        }
        if self
            .enabled
            .iter()
            .any(|pattern| pattern.matches_scope(scope))
        {
            return true;
        }
        self.default.resolve(definition_default)
    }

    pub fn parse(value: &str) -> Self {
        let trimmed = value.trim();
        if trimmed.eq_ignore_ascii_case("all") || trimmed == "*" {
            return Self::all();
        }
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("none") {
            return Self::none();
        }
        Self::only(split_patterns(trimmed).map(|pattern| pattern.to_string()))
    }
}

impl fmt::Display for ProfileRuleSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.default == ProfileDefault::Enabled
            && self.enabled.is_empty()
            && self.disabled.is_empty()
        {
            return f.write_str("all");
        }
        if self.default == ProfileDefault::Disabled
            && self.enabled.is_empty()
            && self.disabled.is_empty()
        {
            return f.write_str("none");
        }
        write!(f, "default={}", self.default)?;
        if !self.enabled.is_empty() {
            write!(f, " enable={}", join_patterns(&self.enabled))?;
        }
        if !self.disabled.is_empty() {
            write!(f, " disable={}", join_patterns(&self.disabled))?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ProfileConfig {
    pub enabled: bool,
    pub label: String,
    pub components: ProfileRuleSet,
    pub scopes: ProfileRuleSet,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

impl ProfileConfig {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            label: "scdata".to_string(),
            components: ProfileRuleSet::from_definitions(),
            scopes: ProfileRuleSet::from_definitions(),
        }
    }

    pub fn enabled(label: impl Into<String>) -> Self {
        Self {
            enabled: true,
            label: label.into(),
            components: ProfileRuleSet::from_definitions(),
            scopes: ProfileRuleSet::from_definitions(),
        }
    }

    pub fn with_global_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    pub fn with_components(mut self, components: ProfileRuleSet) -> Self {
        self.components = components;
        self
    }

    pub fn with_scopes(mut self, scopes: ProfileRuleSet) -> Self {
        self.scopes = scopes;
        self
    }

    pub fn enable_component(mut self, pattern: impl Into<String>) -> Self {
        self.components.enable(pattern);
        self
    }

    pub fn disable_component(mut self, pattern: impl Into<String>) -> Self {
        self.components.disable(pattern);
        self
    }

    pub fn enable_scope(mut self, pattern: impl Into<String>) -> Self {
        self.scopes.enable(pattern);
        self
    }

    pub fn disable_scope(mut self, pattern: impl Into<String>) -> Self {
        self.scopes.disable(pattern);
        self
    }

    pub fn component_enabled(
        &self,
        component: ProfileComponentId,
        definition_default: bool,
    ) -> bool {
        self.enabled
            && self
                .components
                .resolve_component(component, definition_default)
    }

    pub fn scope_enabled(
        &self,
        scope: ProfileScopeId,
        component_default: bool,
        scope_default: bool,
    ) -> bool {
        self.component_enabled(scope.component(), component_default)
            && self.scopes.resolve_scope(scope, scope_default)
    }

    pub fn from_env() -> Self {
        let enabled = env_flag("SCDATA_PROFILE");
        let label = std::env::var("SCDATA_PROFILE_LABEL")
            .ok()
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "scdata".to_string());
        let mut config = Self::enabled(label).with_global_enabled(enabled);

        if let Ok(value) = std::env::var("SCDATA_PROFILE_COMPONENTS") {
            config.components = ProfileRuleSet::parse(&value);
        }
        if let Ok(value) = std::env::var("SCDATA_PROFILE_COMPONENT_ENABLE") {
            for pattern in split_patterns(&value) {
                config.components.enable(pattern.to_string());
            }
        }
        if let Ok(value) = std::env::var("SCDATA_PROFILE_COMPONENT_DISABLE") {
            for pattern in split_patterns(&value) {
                config.components.disable(pattern.to_string());
            }
        }

        if let Ok(value) = std::env::var("SCDATA_PROFILE_SCOPES") {
            config.scopes = ProfileRuleSet::parse(&value);
        }
        if let Ok(value) = std::env::var("SCDATA_PROFILE_SCOPE_ENABLE") {
            for pattern in split_patterns(&value) {
                config.scopes.enable(pattern.to_string());
            }
        }
        if let Ok(value) = std::env::var("SCDATA_PROFILE_SCOPE_DISABLE") {
            for pattern in split_patterns(&value) {
                config.scopes.disable(pattern.to_string());
            }
        }

        config
    }
}

fn join_patterns(patterns: &[ProfilePattern]) -> String {
    let mut joined = String::new();
    for pattern in patterns {
        if !joined.is_empty() {
            joined.push(',');
        }
        joined.push_str(pattern.as_str());
    }
    joined
}

pub(super) fn env_flag(name: &str) -> bool {
    match std::env::var(name) {
        Ok(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "no" | "off"
        ),
        Err(_) => false,
    }
}
