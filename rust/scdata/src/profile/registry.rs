use super::ids::{ProfileComponentId, ProfileScopeId};

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ProfileComponent {
    pub id: ProfileComponentId,
    pub description: &'static str,
    pub default_enabled: bool,
}

impl ProfileComponent {
    pub const fn new(id: ProfileComponentId) -> Self {
        Self {
            id,
            description: "",
            default_enabled: true,
        }
    }

    pub const fn described(mut self, description: &'static str) -> Self {
        self.description = description;
        self
    }

    pub const fn enabled_by_default(mut self) -> Self {
        self.default_enabled = true;
        self
    }

    pub const fn disabled_by_default(mut self) -> Self {
        self.default_enabled = false;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProfileScopeKind {
    Generic,
    Counter,
    Bytes,
    Timer,
    Event,
    Custom(&'static str),
}

impl ProfileScopeKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Generic => "generic",
            Self::Counter => "counter",
            Self::Bytes => "bytes",
            Self::Timer => "timer",
            Self::Event => "event",
            Self::Custom(name) => name,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ProfileScope {
    pub id: ProfileScopeId,
    pub kind: ProfileScopeKind,
    pub description: &'static str,
    pub default_enabled: bool,
}

impl ProfileScope {
    pub const fn new(id: ProfileScopeId) -> Self {
        Self {
            id,
            kind: ProfileScopeKind::Generic,
            description: "",
            default_enabled: true,
        }
    }

    pub const fn described(mut self, description: &'static str) -> Self {
        self.description = description;
        self
    }

    pub const fn kind(mut self, kind: ProfileScopeKind) -> Self {
        self.kind = kind;
        self
    }

    pub const fn enabled_by_default(mut self) -> Self {
        self.default_enabled = true;
        self
    }

    pub const fn disabled_by_default(mut self) -> Self {
        self.default_enabled = false;
        self
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProfileRegistry {
    components: Vec<ProfileComponent>,
    scopes: Vec<ProfileScope>,
}

impl ProfileRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_component(mut self, component: ProfileComponent) -> Self {
        self.add_component(component);
        self
    }

    pub fn with_scope(mut self, scope: ProfileScope) -> Self {
        self.add_scope(scope);
        self
    }

    pub fn add_component(&mut self, component: ProfileComponent) {
        if let Some(existing) = self
            .components
            .iter_mut()
            .find(|existing| existing.id == component.id)
        {
            *existing = component;
        } else {
            self.components.push(component);
        }
    }

    pub fn add_scope(&mut self, scope: ProfileScope) {
        if let Some(existing) = self
            .scopes
            .iter_mut()
            .find(|existing| existing.id == scope.id)
        {
            *existing = scope;
        } else {
            self.scopes.push(scope);
        }
    }

    pub fn components(&self) -> &[ProfileComponent] {
        &self.components
    }

    pub fn scopes(&self) -> &[ProfileScope] {
        &self.scopes
    }
}
