//! Exact numeric distributions with bounded RAM, using streamed external merge sort.
//! Full per-scene CSVs remain the audit source; sort scratch files are temporary.
use super::metrics::NumericDistribution;
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

const RUN_VALUES: usize = 32768;

#[derive(Default)]
struct Values {
    pending: Vec<f64>,
    file: Option<BufWriter<File>>,
    count: usize,
    sum: f64,
}

/// At most RUN_VALUES doubles per metric, two merge read buffers and one write
/// buffer. No list of runs or sample-count-sized quantile arrays is retained.
pub(super) struct NumericCollector {
    values: BTreeMap<String, Values>,
    scratch: PathBuf,
}

impl NumericCollector {
    pub fn new(directory: &Path) -> io::Result<Self> {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        let scratch = directory.join(format!(".numeric-sort-{}-{stamp}", std::process::id()));
        fs::create_dir(&scratch)?;
        Ok(Self {
            values: BTreeMap::new(),
            scratch,
        })
    }

    pub fn push(&mut self, name: &str, value: f64) -> io::Result<()> {
        if !value.is_finite() || !name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_') {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid numeric metric",
            ));
        }
        let values = self.values.entry(name.into()).or_default();
        if values.pending.len() == RUN_VALUES {
            if values.file.is_none() {
                values.file = Some(BufWriter::new(File::create(self.scratch.join(name))?));
            }
            flush_run(values)?;
        }
        values.pending.push(value);
        values.count += 1;
        values.sum += value;
        Ok(())
    }

    pub fn finish(mut self) -> io::Result<BTreeMap<String, NumericDistribution>> {
        let mut result = BTreeMap::new();
        for (name, mut values) in std::mem::take(&mut self.values) {
            if values.file.is_none() {
                result.insert(name, NumericDistribution::from_values(values.pending));
                continue;
            }
            flush_run(&mut values)?;
            values.file.take().unwrap().flush()?;
            let path = merge_runs(self.scratch.join(&name), values.count)?;
            let mut reader = BufReader::new(File::open(&path)?);
            let min = read_value(&mut reader)?;
            reader.seek(SeekFrom::Start((values.count as u64 - 1) * 8))?;
            let max = read_value(&mut reader)?;
            reader.rewind()?;
            let mean = values.sum / values.count as f64;
            let span = (max - min).max(1e-6);
            let mut variance_sum = 0.0;
            let mut bins = vec![0; 20];
            let ranks = [0.05, 0.25, 0.5, 0.75, 0.95]
                .map(|q| ((values.count - 1) as f64 * q).round() as usize);
            let mut percentiles = [0.0; 5];
            for index in 0..values.count {
                let value = read_value(&mut reader)?;
                variance_sum += (value - mean).powi(2);
                bins[(((value - min) / span * 20.0) as usize).min(19)] += 1;
                for (i, rank) in ranks.iter().enumerate() {
                    if index == *rank {
                        percentiles[i] = value;
                    }
                }
            }
            result.insert(
                name,
                NumericDistribution {
                    count: values.count,
                    min,
                    max,
                    mean,
                    standard_deviation: (variance_sum / values.count as f64).sqrt(),
                    percentiles_05_25_50_75_95: percentiles,
                    bin_edges: (0..=20).map(|i| min + span * i as f64 / 20.0).collect(),
                    bin_counts: bins,
                },
            );
        }
        Ok(result)
    }
}

impl Drop for NumericCollector {
    fn drop(&mut self) {
        self.values.clear(); // Close every spool before removing it (also on Windows).
        let _ = fs::remove_dir_all(&self.scratch);
    }
}

fn flush_run(values: &mut Values) -> io::Result<()> {
    values.pending.sort_unstable_by(f64::total_cmp);
    let writer = values.file.as_mut().unwrap();
    for value in values.pending.drain(..) {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn read_value(reader: &mut impl Read) -> io::Result<f64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(f64::from_le_bytes(bytes))
}

fn merge_runs(mut input: PathBuf, count: usize) -> io::Result<PathBuf> {
    let mut output = input.with_extension("merge");
    let mut width = RUN_VALUES;
    while width < count {
        let mut left = BufReader::new(File::open(&input)?);
        let mut right = BufReader::new(File::open(&input)?);
        let mut writer = BufWriter::new(File::create(&output)?);
        let mut start = 0;
        while start < count {
            let mut nleft = width.min(count - start);
            let right_start = start + nleft;
            let mut nright = width.min(count - right_start);
            left.seek(SeekFrom::Start(start as u64 * 8))?;
            right.seek(SeekFrom::Start(right_start as u64 * 8))?;
            let mut a = read_value(&mut left)?;
            let mut b = if nright > 0 {
                read_value(&mut right)?
            } else {
                f64::INFINITY
            };
            let end = right_start + nright;
            while nleft + nright > 0 {
                if nleft > 0 && (nright == 0 || a.total_cmp(&b).is_le()) {
                    writer.write_all(&a.to_le_bytes())?;
                    nleft -= 1;
                    if nleft > 0 {
                        a = read_value(&mut left)?;
                    }
                } else {
                    writer.write_all(&b.to_le_bytes())?;
                    nright -= 1;
                    if nright > 0 {
                        b = read_value(&mut right)?;
                    }
                }
            }
            start = end;
        }
        writer.flush()?;
        std::mem::swap(&mut input, &mut output);
        width = width.saturating_mul(2);
    }
    Ok(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_sort_matches_exact_distribution_across_partial_runs() {
        let directory = tempfile::tempdir().unwrap();
        let mut collector = NumericCollector::new(directory.path()).unwrap();
        let mut input = Vec::new();
        for i in 0..(RUN_VALUES * 5 + 19) {
            let value = ((i * 7919) % 131071) as f64 / 37.0 - 500.0;
            input.push(value);
            collector.push("test", value).unwrap();
            collector.push("constant", 50.0).unwrap();
            assert!(collector.values["test"].pending.len() <= RUN_VALUES);
        }
        let expected = NumericDistribution::from_values(input);
        let result = collector.finish().unwrap();
        let actual = &result["test"];
        assert_eq!(actual.count, expected.count);
        assert_eq!(
            actual.percentiles_05_25_50_75_95,
            expected.percentiles_05_25_50_75_95
        );
        assert_eq!(actual.bin_counts, expected.bin_counts);
        assert_eq!(actual.bin_edges, expected.bin_edges);
        assert!((actual.mean - expected.mean).abs() < 1e-9);
        assert!((actual.standard_deviation - expected.standard_deviation).abs() < 1e-9);
        assert_eq!(result["constant"].standard_deviation, 0.0);
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 0);
    }

    #[test]
    fn unfinished_and_invalid_collectors_clean_scratch() {
        let directory = tempfile::tempdir().unwrap();
        let mut collector = NumericCollector::new(directory.path()).unwrap();
        for _ in 0..=RUN_VALUES {
            collector.push("x", 1.0).unwrap();
        }
        assert!(collector.push("x", f64::NAN).is_err());
        assert!(collector.push("../escape", 1.0).is_err());
        drop(collector);
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 0);
    }
}
