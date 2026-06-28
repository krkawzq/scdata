use std::collections::HashMap;
use std::sync::Arc;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneNameView {
    /// Pointer to UTF-8 bytes owned by the registered dataset's gene table.
    ///
    /// The pointer is valid only while the corresponding dataset remains
    /// registered in the `DataBank`. It may dangle after `DataBank::unregister`
    /// or after `DataBank` is dropped.
    pub ptr: *const u8,
    /// Byte length of the UTF-8 gene name.
    pub len: usize,
}

// SAFETY: `GeneNameView` is a plain pointer/length view into bytes owned by
// `DatasetGeneRefs.names` (`Arc<str>`). Sharing or sending the view is sound as
// long as the corresponding dataset stays registered, which is the same
// lifetime contract required by the existing FFI-facing view API.
unsafe impl Send for GeneNameView {}
unsafe impl Sync for GeneNameView {}

#[derive(Debug, Default)]
pub struct GeneInterner {
    strings: HashMap<String, InternedGene>,
}

#[derive(Debug)]
struct InternedGene {
    value: Arc<str>,
    refcount: usize,
}

#[derive(Debug, Clone)]
pub struct DatasetGeneRefs {
    names: Vec<Arc<str>>,
    views: Vec<GeneNameView>,
}

impl GeneInterner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn intern_dataset(&mut self, names: &[String]) -> DatasetGeneRefs {
        let mut interned = Vec::with_capacity(names.len());
        let mut views = Vec::with_capacity(names.len());

        for name in names {
            let value = if let Some(existing) = self.strings.get_mut(name.as_str()) {
                existing.refcount += 1;
                Arc::clone(&existing.value)
            } else {
                let value: Arc<str> = Arc::from(name.as_str());
                self.strings.insert(
                    name.clone(),
                    InternedGene {
                        value: Arc::clone(&value),
                        refcount: 1,
                    },
                );
                value
            };
            views.push(GeneNameView {
                ptr: value.as_ptr(),
                len: value.len(),
            });
            interned.push(value);
        }

        DatasetGeneRefs {
            names: interned,
            views,
        }
    }

    pub fn release_dataset(&mut self, refs: &DatasetGeneRefs) {
        let mut remove = Vec::new();
        for name in &refs.names {
            let Some(entry) = self.strings.get_mut(name.as_ref()) else {
                continue;
            };
            entry.refcount = entry.refcount.saturating_sub(1);
            if entry.refcount == 0 {
                remove.push(name.to_string());
            }
        }
        for name in remove {
            self.strings.remove(&name);
        }
    }
}

impl GeneNameView {
    pub const fn empty() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }

    pub fn is_empty(self) -> bool {
        self.len == 0
    }
}

impl DatasetGeneRefs {
    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn views(&self) -> &[GeneNameView] {
        &self.views
    }

    pub(crate) fn names(&self) -> &[Arc<str>] {
        &self.names
    }
}
