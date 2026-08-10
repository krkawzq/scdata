mod common;

use sc_compress::DenseWriter;
use scdata::{
    compile, Dataset, Error, FeatureMap, Fill, FloatCastPolicy, OutputDType, OutputSpec,
    OverflowPolicy, PlanSpec, RowRef, SessionState, Source, SourceId,
};

use common::{blocking, drain_rows};

#[test]
fn public_validation_rejects_invalid_specs_before_execution() {
    assert!(matches!(
        FeatureMap::new([Some(0), Some(0)]),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        FeatureMap::from_signed(&[-2, 0]),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        OutputSpec::new(1, OutputDType::F32, Fill::U16(0)),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        compile(PlanSpec::new(
            vec![],
            vec![],
            OutputSpec::new(1, OutputDType::U16, Fill::U16(0)).unwrap(),
            0,
            2,
        )),
        Err(Error::InvalidConfig(_))
    ));

    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("validation");
    DenseWriter::new(&path).write(&[1u16, 2], [1, 2]).unwrap();
    let dataset = Dataset::open(&path).unwrap();
    let output = OutputSpec::new(2, OutputDType::U16, Fill::U16(0)).unwrap();

    let wrong_map =
        Source::new(0, dataset.clone()).feature_map(FeatureMap::new([Some(0)]).unwrap());
    assert!(matches!(
        compile(PlanSpec::new(
            vec![wrong_map],
            vec![RowRef::new(SourceId::new(0), 0)],
            output.clone(),
            1,
            2,
        )),
        Err(Error::InvalidInput(_))
    ));

    assert!(matches!(
        compile(PlanSpec::new(
            vec![Source::new(0, dataset.clone())],
            vec![RowRef::new(SourceId::new(9), 0)],
            output.clone(),
            1,
            2,
        )),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        compile(PlanSpec::new(
            vec![Source::new(0, dataset)],
            vec![RowRef::new(SourceId::new(0), 1)],
            output,
            1,
            2,
        )),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn overflow_policy_and_batch_views_follow_the_public_contract() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("overflow");
    DenseWriter::new(&path)
        .write(&[-3i16, 4, 5, 6], [2, 2])
        .unwrap();
    let output = OutputSpec::new(2, OutputDType::U16, Fill::U16(99))
        .unwrap()
        .overflow(OverflowPolicy::UseValue(Fill::U16(777)))
        .unwrap();
    let plan = compile(PlanSpec::new(
        vec![Source::new(0, Dataset::open(&path).unwrap())],
        vec![
            RowRef::new(SourceId::new(0), 0),
            RowRef::new(SourceId::new(0), 1),
        ],
        output,
        2,
        2,
    ))
    .unwrap();
    let mut session = plan.open(blocking(1)).unwrap();
    let batch = session.next_batch().unwrap().unwrap();

    assert_eq!(batch.rows(), 2);
    assert_eq!(batch.n_cols(), 2);
    assert_eq!(batch.dtype(), OutputDType::U16);
    assert_eq!(batch.row_as::<u16>(0).unwrap(), &[777, 4]);
    assert_eq!(batch.row_as::<u16>(1).unwrap(), &[5, 6]);
    assert!(matches!(
        batch.row_as::<i16>(0),
        Err(Error::InvalidInput(_))
    ));
    assert!(batch.row(2).is_none());
    assert!(matches!(
        batch.as_slice::<u16>(),
        Err(Error::Unsupported(_))
    ));
    let padded = batch.as_padded_slice::<u16>().unwrap();
    assert_eq!(&padded[..2], &[777, 4]);
    assert!(padded[2..32].iter().all(|value| *value == 0));
    assert_eq!(&padded[32..34], &[5, 6]);
    assert!(padded[34..64].iter().all(|value| *value == 0));

    drop(batch);
    assert!(session.next_batch().unwrap().is_none());
    assert_eq!(session.state(), SessionState::Finished);
}

#[test]
fn potentially_rounding_float_cast_requires_explicit_opt_in() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("rounding");
    DenseWriter::new(&path)
        .write(&[16_777_217i32], [1, 1])
        .unwrap();
    let dataset = Dataset::open(&path).unwrap();
    let rows = vec![RowRef::new(SourceId::new(0), 0)];

    assert!(matches!(
        compile(PlanSpec::new(
            vec![Source::new(0, dataset.clone())],
            rows.clone(),
            OutputSpec::new(1, OutputDType::F32, Fill::F32(0.0)).unwrap(),
            1,
            2,
        )),
        Err(Error::Promote(_))
    ));

    let output = OutputSpec::new(1, OutputDType::F32, Fill::F32(0.0))
        .unwrap()
        .float_cast(FloatCastPolicy::AllowRounding);
    let plan = compile(PlanSpec::new(
        vec![Source::new(0, dataset)],
        rows,
        output,
        1,
        2,
    ))
    .unwrap();
    assert_eq!(drain_rows::<f32>(&plan, 1).0, vec![vec![16_777_216.0]]);
}

#[test]
fn empty_plan_opens_as_an_already_finished_session() {
    let plan = compile(PlanSpec::new(
        vec![],
        vec![],
        OutputSpec::new(3, OutputDType::F32, Fill::F32(0.0)).unwrap(),
        4,
        2,
    ))
    .unwrap();
    assert!(plan.is_empty());
    assert_eq!(plan.batch_count(), 0);
    assert_eq!(plan.stats().output_ring_bytes, 0);

    let mut session = plan.open(blocking(1)).unwrap();
    assert_eq!(session.state(), SessionState::Finished);
    assert!(session.next_batch().unwrap().is_none());
}
