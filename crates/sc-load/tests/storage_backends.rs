mod common;

use std::io::Write;

use sc_compress::{DenseWriter, Partition};
use sc_load::{
    compile, Dataset, Error, Fill, IoMode, OutputDType, OutputSpec, PlanSpec, RowRef, Source,
    SourceId, StoreLocation,
};
use zip::write::SimpleFileOptions;

use common::{drain_rows, session_config};

#[test]
fn zip_sources_execute_through_public_store_locations() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("dense");
    DenseWriter::new(&root, Partition::fixed_cells(2), Partition::fixed_cells(1))
        .write(&[1u16, 2, 3, 4], [2, 2])
        .unwrap();

    for method in [
        zip::CompressionMethod::Stored,
        zip::CompressionMethod::Deflated,
    ] {
        let archive = temporary.path().join(format!("{method:?}.zip"));
        write_zip(&root, &archive, method);
        let plan = compile(PlanSpec::new(
            vec![Source::new(
                0,
                Dataset::open(StoreLocation::zip(&archive, "assay")).unwrap(),
            )],
            vec![RowRef::new(SourceId::new(0), 1)],
            OutputSpec::new(2, OutputDType::U16, Fill::U16(0)).unwrap(),
            1,
            2,
        ))
        .unwrap();
        assert_eq!(drain_rows::<u16>(&plan, 1).0, vec![vec![3, 4]]);
    }
}

#[test]
fn deflated_zip_forces_auto_to_blocking_and_rejects_explicit_uring() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("dense");
    DenseWriter::new(&root, Partition::fixed_cells(2), Partition::fixed_cells(1))
        .write(&[1u16, 2, 3, 4], [2, 2])
        .unwrap();
    let archive = temporary.path().join("deflated.zip");
    write_zip(&root, &archive, zip::CompressionMethod::Deflated);
    let plan = compile(PlanSpec::new(
        vec![Source::new(
            0,
            Dataset::open(StoreLocation::zip(&archive, "assay")).unwrap(),
        )],
        vec![RowRef::new(SourceId::new(0), 0)],
        OutputSpec::new(2, OutputDType::U16, Fill::U16(0)).unwrap(),
        1,
        2,
    ))
    .unwrap();

    assert!(matches!(
        plan.open(session_config(1, IoMode::Uring { queue_depth: 2 })),
        Err(Error::Unsupported(_))
    ));

    let mut session = plan
        .open(session_config(1, IoMode::Auto { queue_depth: 2 }))
        .unwrap();
    assert_eq!(session.stats().actual_io_mode, IoMode::Blocking);
    assert_eq!(
        session
            .next_batch()
            .unwrap()
            .unwrap()
            .row_as::<u16>(0)
            .unwrap(),
        &[1, 2]
    );
}

fn write_zip(root: &std::path::Path, archive: &std::path::Path, method: zip::CompressionMethod) {
    let file = std::fs::File::create(archive).unwrap();
    let mut writer = zip::ZipWriter::new(file);
    let options = SimpleFileOptions::default().compression_method(method);
    for key in ["meta.json", "data/0"] {
        writer.start_file(format!("assay/{key}"), options).unwrap();
        writer
            .write_all(&std::fs::read(root.join(key)).unwrap())
            .unwrap();
    }
    writer.finish().unwrap();
}
