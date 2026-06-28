use super::dataset::Dataset;
use super::error::{DataBankError, DataBankResult};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DatasetId {
    pub slot: u32,
    pub generation: u32,
}

#[derive(Debug)]
struct DatasetSlot {
    generation: u32,
    dataset: Option<Arc<Dataset>>,
}

#[derive(Debug, Default)]
pub struct DatasetRegistry {
    slots: Vec<DatasetSlot>,
    free_slots: Vec<u32>,
}

impl DatasetRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn ensure_can_register(&self) -> DataBankResult<()> {
        if self.free_slots.is_empty() && self.slots.len() > u32::MAX as usize {
            return Err(DataBankError::InvalidConfig(
                "dataset slot count exceeds u32".to_string(),
            ));
        }
        Ok(())
    }

    pub fn register(&mut self, dataset: Dataset) -> DataBankResult<DatasetId> {
        let dataset = Arc::new(dataset);
        if let Some(slot) = self.free_slots.pop() {
            let entry =
                self.slots
                    .get_mut(slot as usize)
                    .ok_or(DataBankError::InvalidDatasetId(DatasetId {
                        slot,
                        generation: 0,
                    }))?;
            entry.generation = next_generation(entry.generation);
            entry.dataset = Some(dataset);
            return Ok(DatasetId {
                slot,
                generation: entry.generation,
            });
        }

        let slot = u32::try_from(self.slots.len()).map_err(|_| {
            DataBankError::InvalidConfig("dataset slot count exceeds u32".to_string())
        })?;
        let generation = 1;
        self.slots.push(DatasetSlot {
            generation,
            dataset: Some(dataset),
        });
        Ok(DatasetId { slot, generation })
    }

    pub fn get(&self, id: DatasetId) -> DataBankResult<&Dataset> {
        self.slot(id)?
            .dataset
            .as_ref()
            .map(Arc::as_ref)
            .ok_or(DataBankError::DatasetUnloaded(id))
    }

    pub fn get_arc(&self, id: DatasetId) -> DataBankResult<Arc<Dataset>> {
        self.slot(id)?
            .dataset
            .as_ref()
            .cloned()
            .ok_or(DataBankError::DatasetUnloaded(id))
    }

    pub fn remove(&mut self, id: DatasetId) -> DataBankResult<Arc<Dataset>> {
        let slot = self.slot_mut(id)?;
        let dataset = slot
            .dataset
            .take()
            .ok_or(DataBankError::DatasetUnloaded(id))?;
        self.free_slots.push(id.slot);
        Ok(dataset)
    }

    pub fn drain(&mut self) -> Vec<Arc<Dataset>> {
        self.free_slots.clear();
        self.slots
            .iter_mut()
            .filter_map(|slot| slot.dataset.take())
            .collect()
    }

    fn slot(&self, id: DatasetId) -> DataBankResult<&DatasetSlot> {
        let slot = self
            .slots
            .get(id.slot as usize)
            .ok_or(DataBankError::InvalidDatasetId(id))?;
        if slot.generation != id.generation {
            return Err(DataBankError::InvalidDatasetId(id));
        }
        Ok(slot)
    }

    fn slot_mut(&mut self, id: DatasetId) -> DataBankResult<&mut DatasetSlot> {
        let slot = self
            .slots
            .get_mut(id.slot as usize)
            .ok_or(DataBankError::InvalidDatasetId(id))?;
        if slot.generation != id.generation {
            return Err(DataBankError::InvalidDatasetId(id));
        }
        Ok(slot)
    }
}

fn next_generation(current: u32) -> u32 {
    current.wrapping_add(1).max(1)
}
