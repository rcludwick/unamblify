// SPDX-License-Identifier: AGPL-3.0-only
// Licensed under the GNU Affero General Public License v3.0 only. See LICENSE.

//! Rows written before the GPU fields existed must still load: the
//! history file is a day of accumulated readings, and dropping it on an
//! upgrade would defeat the point of persisting it at all.

use std::io::Write;

use unamblify_web::persist;

#[test]
fn rows_without_gpu_fields_still_load() {
    let dir = tempfile::tempdir().unwrap();
    let p = persist::path(dir.path());
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    // Verbatim shape of the rows already on disk.
    let old = r#"{"t":1789755527,"cpu":17.055584,"load1":3.4375,"temps":{"/dev/disk0":57,"/dev/disk8":58}}"#;
    let mut f = std::fs::File::create(&p).unwrap();
    writeln!(f, "{old}").unwrap();
    drop(f);

    let rows = persist::load(&p, 0);
    assert_eq!(rows.len(), 1, "the pre-GPU row was dropped");
    assert_eq!(rows[0].t, 1_789_755_527);
    assert_eq!(rows[0].gpu, None);
    assert_eq!(rows[0].gpu_mem_gb, None);
    assert_eq!(rows[0].temps["/dev/disk8"], 58);
    assert!(
        rows[0].io.is_empty(),
        "a row from before the I/O rates loads with none"
    );
}
