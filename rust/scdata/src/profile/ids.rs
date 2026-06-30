use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ProfileComponentId {
    name: &'static str,
}

impl ProfileComponentId {
    pub const fn new(name: &'static str) -> Self {
        Self { name }
    }

    pub const fn as_str(self) -> &'static str {
        self.name
    }

    pub fn matches(self, pattern: &ProfilePattern) -> bool {
        pattern.matches_component(self)
    }
}

impl fmt::Display for ProfileComponentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ProfileScopeId {
    component: ProfileComponentId,
    name: &'static str,
}

impl ProfileScopeId {
    pub const fn new(component: ProfileComponentId, name: &'static str) -> Self {
        Self { component, name }
    }

    pub const fn component(self) -> ProfileComponentId {
        self.component
    }

    pub const fn name(self) -> &'static str {
        self.name
    }

    pub fn full_name(self) -> String {
        format!("{}.{}", self.component, self.name)
    }

    pub fn matches(self, pattern: &ProfilePattern) -> bool {
        pattern.matches_scope(self)
    }
}

impl fmt::Display for ProfileScopeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.component, self.name)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProfilePattern {
    normalized: String,
}

impl ProfilePattern {
    pub fn new(pattern: impl Into<String>) -> Self {
        Self {
            normalized: normalize_pattern(&pattern.into()),
        }
    }

    pub fn as_str(&self) -> &str {
        &self.normalized
    }

    pub fn is_all(&self) -> bool {
        self.normalized == "all" || self.normalized == "*"
    }

    pub fn matches_component(&self, component: ProfileComponentId) -> bool {
        if self.is_all() {
            return true;
        }
        let component = normalize_pattern(component.as_str());
        if self.normalized == component {
            return true;
        }
        if let Some(prefix) = self.normalized.strip_suffix('*') {
            return component.starts_with(prefix);
        }
        false
    }

    pub fn matches_scope(&self, scope: ProfileScopeId) -> bool {
        if self.is_all() {
            return true;
        }
        let component = normalize_pattern(scope.component().as_str());
        let full = normalize_pattern(&scope.full_name());
        if self.normalized == component || self.normalized == full {
            return true;
        }
        if let Some(prefix) = self.normalized.strip_suffix(".*") {
            return full.starts_with(prefix) && full[prefix.len()..].starts_with('.');
        }
        if let Some(prefix) = self.normalized.strip_suffix('*') {
            return full.starts_with(prefix);
        }
        false
    }
}

impl fmt::Display for ProfilePattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.normalized)
    }
}

pub(super) fn normalize_pattern(pattern: &str) -> String {
    pattern.trim().to_ascii_lowercase().replace('_', "-")
}

pub(super) fn split_patterns(value: &str) -> impl Iterator<Item = ProfilePattern> + '_ {
    value
        .split(|ch: char| ch == ',' || ch == ';' || ch.is_ascii_whitespace())
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(ProfilePattern::new)
}
