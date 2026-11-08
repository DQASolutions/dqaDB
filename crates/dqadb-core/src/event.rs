//! Event files: the source of truth. One immutable JSON file per event at
//! `events/YYYY/MM/DD/<bucket>/<ULID>.json` (spec §5).

use chrono::{DateTime, Datelike, SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use ulid::Ulid;

use crate::rdf::Term;
use crate::{Error, Result};

pub const EVENT_SCHEMA: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub dqadb: u32,
    pub event_id: String,
    pub ts: String,
    pub actor: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<Source>,
    pub ontology: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    pub ops: Vec<Op>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpKind {
    Add,
    Retract,
}

/// A stored op. Subjects and predicates are full IRIs and objects are normalised
/// terms, so replay never depends on the prefix map that was active at write time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Op {
    pub op: OpKind,
    pub s: String,
    pub p: String,
    pub o: Term,
}

impl Event {
    pub fn ulid(&self) -> Result<Ulid> {
        parse_ulid(&self.event_id)
    }
}

pub fn parse_ulid(s: &str) -> Result<Ulid> {
    Ulid::from_string(s).map_err(|e| Error::other(format!("invalid ULID '{s}': {e}")))
}

pub fn ulid_datetime(id: Ulid) -> DateTime<Utc> {
    DateTime::<Utc>::from_timestamp_millis(id.timestamp_ms() as i64).unwrap_or_default()
}

pub fn ulid_ts(id: Ulid) -> String {
    ulid_datetime(id).to_rfc3339_opts(SecondsFormat::Millis, true)
}

type Day = (i32, u32, u32);

fn ulid_day(id: Ulid) -> Day {
    let dt = ulid_datetime(id);
    (dt.year(), dt.month(), dt.day())
}

/// Bucket = characters 11-12 of the ULID (first two random characters): 1024 buckets per day.
pub fn bucket(id: &str) -> &str {
    &id[10..12]
}

pub fn rel_path(id: Ulid) -> PathBuf {
    let s = id.to_string();
    let (y, m, d) = ulid_day(id);
    PathBuf::from(format!("{y:04}/{m:02}/{d:02}/{}/{s}.json", bucket(&s)))
}

/// Canonical bytes: keys sorted (serde_json's default map is ordered), no extra whitespace.
pub fn canonical_bytes(ev: &Event) -> Result<Vec<u8>> {
    let value = serde_json::to_value(ev)?;
    let mut bytes = serde_json::to_vec(&value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

/// Write the event atomically: tmp file -> fsync -> rename -> fsync dir.
pub fn write_atomic(events_dir: &Path, ev: &Event) -> Result<PathBuf> {
    let id = ev.ulid()?;
    let path = events_dir.join(rel_path(id));
    let dir = path.parent().expect("event path has a parent");
    fs::create_dir_all(dir)?;
    if path.exists() {
        return Err(Error::other(format!(
            "event file already exists: {}",
            path.display()
        )));
    }
    let tmp = dir.join(format!(".tmp-{}", ev.event_id));
    {
        let mut f = File::create(&tmp)?;
        f.write_all(&canonical_bytes(ev)?)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, &path)?;
    File::open(dir)?.sync_all()?;
    Ok(path)
}

pub fn read_event(path: &Path) -> Result<Event> {
    let bytes = fs::read(path)?;
    serde_json::from_slice(&bytes)
        .map_err(|e| Error::other(format!("cannot parse event {}: {e}", path.display())))
}

pub fn find_event(events_dir: &Path, id: Ulid) -> Result<PathBuf> {
    let path = events_dir.join(rel_path(id));
    if path.exists() {
        Ok(path)
    } else {
        Err(Error::NotFound(format!("event {id}")))
    }
}

// ---------- directory walking ----------

fn numeric_children(dir: &Path) -> Result<Vec<(u32, PathBuf)>> {
    let mut out = Vec::new();
    if !dir.exists() {
        return Ok(out);
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        if let Some(n) = entry
            .file_name()
            .to_str()
            .and_then(|s| s.parse::<u32>().ok())
        {
            out.push((n, entry.path()));
        }
    }
    out.sort();
    Ok(out)
}

/// All day directories, ascending.
pub fn day_dirs(events_dir: &Path) -> Result<Vec<(Day, PathBuf)>> {
    let mut out = Vec::new();
    for (y, ydir) in numeric_children(events_dir)? {
        for (m, mdir) in numeric_children(&ydir)? {
            for (d, ddir) in numeric_children(&mdir)? {
                out.push(((y as i32, m, d), ddir));
            }
        }
    }
    Ok(out)
}

/// Event files of one day (all buckets), sorted by ULID.
pub fn day_files(day_dir: &Path) -> Result<Vec<(Ulid, PathBuf)>> {
    let mut out = Vec::new();
    for bucket in fs::read_dir(day_dir)? {
        let bucket = bucket?;
        if !bucket.file_type()?.is_dir() {
            continue;
        }
        for f in fs::read_dir(bucket.path())? {
            let f = f?;
            let name = f.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some(stem) = name.strip_suffix(".json") else {
                continue;
            };
            if stem.starts_with('.') {
                continue;
            }
            if let Ok(id) = Ulid::from_string(stem) {
                out.push((id, f.path()));
            }
        }
    }
    out.sort();
    Ok(out)
}

/// Event files with ULID strictly greater than `after` (all if None), ascending.
/// Only day directories on or after the watermark's day are scanned.
pub fn list_after(events_dir: &Path, after: Option<Ulid>) -> Result<Vec<(Ulid, PathBuf)>> {
    let min_day = after.map(ulid_day);
    let mut out = Vec::new();
    for (day, dir) in day_dirs(events_dir)? {
        if min_day.is_some_and(|m| day < m) {
            continue;
        }
        out.extend(
            day_files(&dir)?
                .into_iter()
                .filter(|(id, _)| after.is_none_or(|a| *id > a)),
        );
    }
    Ok(out)
}

/// Newest event id on disk (scans only the newest non-empty day).
pub fn latest(events_dir: &Path) -> Result<Option<Ulid>> {
    for (_, dir) in day_dirs(events_dir)?.into_iter().rev() {
        if let Some((id, _)) = day_files(&dir)?.into_iter().next_back() {
            return Ok(Some(id));
        }
    }
    Ok(None)
}

/// Newest-first iterator over event files, optionally stopping before `since` (inclusive bound).
pub fn list_recent(
    events_dir: &Path,
    since: Option<Day>,
) -> Result<impl Iterator<Item = Result<(Ulid, PathBuf)>>> {
    let days: Vec<_> = day_dirs(events_dir)?
        .into_iter()
        .rev()
        .take_while(move |(day, _)| since.is_none_or(|s| *day >= s))
        .collect();
    Ok(days.into_iter().flat_map(|(_, dir)| match day_files(&dir) {
        Ok(files) => files.into_iter().rev().map(Ok).collect::<Vec<_>>(),
        Err(e) => vec![Err(e)],
    }))
}

pub fn parse_day(s: &str) -> Result<Day> {
    let d = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d")
        .map_err(|_| Error::other(format!("invalid date '{s}', expected YYYY-MM-DD")))?;
    Ok((d.year(), d.month(), d.day()))
}
