use std::ops::Range;

use crate::error::{vector_with_capacity, Error, Result};

/// Split contiguous work into non-empty ranges with approximately equal weight.
pub(crate) fn balanced_ranges(
    item_count: usize,
    total_weight: usize,
    maximum_ranges: usize,
    weight_at: impl Fn(usize) -> usize,
) -> Result<Vec<Range<usize>>> {
    if item_count == 0 {
        return Ok(Vec::new());
    }
    if maximum_ranges == 0 {
        return Err(Error::InvalidArgument(
            "parallel range count must be non-zero".into(),
        ));
    }

    let range_count = item_count.min(maximum_ranges);
    let mut ranges = vector_with_capacity(range_count)?;
    let mut first = 0usize;
    let mut boundary = 0usize;
    let mut cumulative_weight = 0usize;
    for completed_ranges in 1..range_count {
        let target =
            ((total_weight as u128 * completed_ranges as u128) / range_count as u128) as usize;
        let maximum_boundary = item_count - (range_count - completed_ranges);
        while boundary < maximum_boundary {
            let candidate_weight = cumulative_weight
                .checked_add(weight_at(boundary))
                .ok_or_else(|| Error::InvalidArgument("parallel work weight overflow".into()))?;
            if boundary == first
                || candidate_weight.abs_diff(target) <= cumulative_weight.abs_diff(target)
            {
                cumulative_weight = candidate_weight;
                boundary += 1;
            } else {
                break;
            }
        }
        ranges.push(first..boundary);
        first = boundary;
    }
    ranges.push(first..item_count);

    debug_assert_eq!(ranges.len(), range_count);
    Ok(ranges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skewed_weights_are_split_by_work_instead_of_item_count() {
        let weights = [100, 100, 100, 100, 10, 10, 10, 10];
        let ranges = balanced_ranges(weights.len(), weights.iter().sum(), 4, |index| {
            weights[index]
        })
        .unwrap();

        assert_eq!(ranges, [0..1, 1..2, 2..3, 3..8]);
        assert_eq!(ranges.first().unwrap().start, 0);
        assert_eq!(ranges.last().unwrap().end, weights.len());
        assert!(ranges.windows(2).all(|pair| pair[0].end == pair[1].start));
    }

    #[test]
    fn ranges_cover_random_weight_sequences_exactly() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for item_count in 1..=64 {
            let weights = (0..item_count)
                .map(|_| {
                    state ^= state >> 12;
                    state ^= state << 25;
                    state ^= state >> 27;
                    (state as usize % 4096) + 1
                })
                .collect::<Vec<_>>();
            for maximum_ranges in 1..=item_count + 2 {
                let ranges =
                    balanced_ranges(item_count, weights.iter().sum(), maximum_ranges, |index| {
                        weights[index]
                    })
                    .unwrap();
                assert_eq!(ranges.len(), item_count.min(maximum_ranges));
                assert_eq!(ranges.first().unwrap().start, 0);
                assert_eq!(ranges.last().unwrap().end, item_count);
                assert!(ranges.iter().all(|range| !range.is_empty()));
                assert!(ranges.windows(2).all(|pair| pair[0].end == pair[1].start));
            }
        }
    }
}
