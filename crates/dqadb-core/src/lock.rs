use std::fs::{File, OpenOptions, TryLockError};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::{Error, Result};

/// Exclusive write lock on `events/.lock`. Released when dropped (the file is closed).
/// Only one process may append events or update the index at a time; readers never take it.
pub struct WriteLock {
    _file: File,
    pub path: PathBuf,
}

impl WriteLock {
    pub fn acquire(path: &Path, timeout: Duration) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(path)?;
        let start = Instant::now();
        loop {
            match file.try_lock() {
                Ok(()) => {
                    return Ok(Self {
                        _file: file,
                        path: path.to_path_buf(),
                    });
                }
                Err(TryLockError::WouldBlock) => {
                    if start.elapsed() >= timeout {
                        return Err(Error::LockTimeout(path.to_path_buf()));
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(TryLockError::Error(e)) => return Err(e.into()),
            }
        }
    }
}
