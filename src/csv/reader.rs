use crate::array::{variable, Array, BinaryBuilder, Builder, PrimitiveBuilder, StringBuilder};
use crate::datatypes::*;
use crate::memory::AllocationError;
use crate::record;
use csv_core::ReadRecordResult;
use dashmap::DashMap;
use std::collections::HashMap;
use std::fmt;
use std::io::{BufRead, BufReader, Read};
use std::str::{self, FromStr};
use std::sync::{Arc, Mutex};

pub struct Record {
    fields: Vec<u8>,
    ends: Vec<usize>,
}

impl Record {
    #[must_use]
    pub fn from_data(data: &[&[u8]]) -> Vec<Self> {
        let mut reader = csv_core::Reader::new();
        data.iter()
            .filter_map(|d| Self::new(&mut reader, d))
            .collect()
    }

    /// # Panics
    ///
    /// Panics if `input.len() * 2` overflows `usize`.
    ///
    #[must_use]
    pub fn new(reader: &mut csv_core::Reader, input: &[u8]) -> Option<Self> {
        let mut fields = Vec::with_capacity(input.len());
        let mut ends = Vec::with_capacity(input.len());
        let mut cur = 0;
        let (mut outlen, mut endlen) = (0, 0);
        loop {
            let (res, nin, nout, nend) =
                reader.read_record(&input[cur..], &mut fields[outlen..], &mut ends[endlen..]);
            cur += nin;
            outlen += nout;
            endlen += nend;
            match res {
                ReadRecordResult::InputEmpty => continue,
                ReadRecordResult::OutputFull => {
                    fields.resize(std::cmp::max(4, fields.len().checked_mul(2).unwrap()), 0)
                }
                ReadRecordResult::OutputEndsFull => {
                    ends.resize(std::cmp::max(4, ends.len().checked_mul(2).unwrap()), 0)
                }
                ReadRecordResult::Record => {
                    unsafe {
                        fields.set_len(outlen);
                        ends.set_len(endlen);
                    }
                    return Some(Self { fields, ends });
                }
                ReadRecordResult::End => return None,
            }
        }
    }

    /// # Panics
    ///
    /// Panics if line length in input * 2 overflows `usize`.
    ///
    #[must_use]
    pub fn from_buf(reader: &mut csv_core::Reader, input: &mut dyn BufRead) -> Option<Self> {
        let mut fields = Vec::with_capacity(1024);
        let mut ends = Vec::with_capacity(1024);
        let (mut outlen, mut endlen) = (0, 0);
        loop {
            let (res, nin, nout, nend) = {
                let buf = input.fill_buf().expect("file reading error");
                reader.read_record(buf, &mut fields[outlen..], &mut ends[endlen..])
            };
            input.consume(nin);
            outlen += nout;
            endlen += nend;
            match res {
                ReadRecordResult::InputEmpty => continue,
                ReadRecordResult::OutputFull => {
                    fields.resize(std::cmp::max(4, fields.len().checked_mul(2).unwrap()), 0)
                }
                ReadRecordResult::OutputEndsFull => {
                    ends.resize(std::cmp::max(4, ends.len().checked_mul(2).unwrap()), 0)
                }
                ReadRecordResult::Record => {
                    unsafe {
                        fields.set_len(outlen);
                        ends.set_len(endlen);
                    }
                    return Some(Self { fields, ends });
                }
                ReadRecordResult::End => return None,
            }
        }
    }

    #[inline]
    #[must_use]
    pub fn get(&self, i: usize) -> Option<&[u8]> {
        let end = match self.ends.get(i) {
            None => return None,
            Some(&end) => end,
        };
        let start = match i.checked_sub(1).and_then(|i| self.ends.get(i)) {
            None => 0,
            Some(&start) => start,
        };
        Some(&self.fields[start..end])
    }
}

pub struct ParseError {
    inner: Box<dyn std::error::Error>,
}

impl fmt::Debug for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "parse error: {}", self.inner)
    }
}

impl From<std::net::AddrParseError> for ParseError {
    fn from(error: std::net::AddrParseError) -> Self {
        Self {
            inner: Box::new(error),
        }
    }
}

impl From<std::num::ParseFloatError> for ParseError {
    fn from(error: std::num::ParseFloatError) -> Self {
        Self {
            inner: Box::new(error),
        }
    }
}

impl From<std::num::ParseIntError> for ParseError {
    fn from(error: std::num::ParseIntError) -> Self {
        Self {
            inner: Box::new(error),
        }
    }
}

impl From<chrono::format::ParseError> for ParseError {
    fn from(error: chrono::format::ParseError) -> Self {
        Self {
            inner: Box::new(error),
        }
    }
}

impl From<std::str::Utf8Error> for ParseError {
    fn from(error: std::str::Utf8Error) -> Self {
        Self {
            inner: Box::new(error),
        }
    }
}

pub type Int64Parser = dyn Fn(&[u8]) -> Result<i64, ParseError> + Send + Sync;
pub type UInt32Parser = dyn Fn(&[u8]) -> Result<u32, ParseError> + Send + Sync;
pub type Float64Parser = dyn Fn(&[u8]) -> Result<f64, ParseError> + Send + Sync;

#[derive(Clone)]
pub enum FieldParser {
    Int64(Arc<Int64Parser>),
    UInt32(Arc<UInt32Parser>),
    Float64(Arc<Float64Parser>),
    Utf8,
    Binary,
    Timestamp(Arc<Int64Parser>),
    Dict,
}

impl FieldParser {
    #[must_use]
    pub fn int64() -> Self {
        Self::Int64(Arc::new(parse::<i64>))
    }

    #[must_use]
    pub fn uint32() -> Self {
        Self::UInt32(Arc::new(parse::<u32>))
    }

    #[must_use]
    pub fn float64() -> Self {
        Self::Float64(Arc::new(parse::<f64>))
    }

    #[must_use]
    pub fn timestamp() -> Self {
        Self::Int64(Arc::new(parse_timestamp))
    }

    #[must_use]
    pub fn uint32_with_parser<P>(parser: P) -> Self
    where
        P: Fn(&[u8]) -> Result<u32, ParseError> + Send + Sync + 'static,
    {
        Self::UInt32(Arc::new(parser))
    }

    #[must_use]
    pub fn timestamp_with_parser<P>(parser: P) -> Self
    where
        P: Fn(&[u8]) -> Result<i64, ParseError> + Send + Sync + 'static,
    {
        Self::Timestamp(Arc::new(parser))
    }
}

impl<'a> fmt::Debug for FieldParser {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int64(_) => write!(f, "Int64"),
            Self::UInt32(_) => write!(f, "UInt32"),
            Self::Float64(_) => write!(f, "Float64"),
            Self::Utf8 => write!(f, "Utf8"),
            Self::Binary => write!(f, "Binary"),
            Self::Timestamp(_) => write!(f, "Timestamp"),
            Self::Dict => write!(f, "Dict"),
        }
    }
}

fn parse<T>(v: &[u8]) -> Result<T, ParseError>
where
    T: FromStr,
    <T as FromStr>::Err: Into<ParseError>,
{
    std::str::from_utf8(v)?.parse::<T>().map_err(Into::into)
}

/// Parses timestamp in RFC 3339 format.
fn parse_timestamp(v: &[u8]) -> Result<i64, ParseError> {
    Ok(
        chrono::NaiveDateTime::parse_from_str(str::from_utf8(v)?, "%Y-%m-%dT%H:%M:%S%.f%:z")?
            .timestamp(),
    )
}

type ConcurrentEnumMaps = Arc<DashMap<usize, Arc<DashMap<String, (u32, usize)>>>>;

/// CSV reader
pub struct Reader<'a, I, H>
where
    I: Iterator<Item = &'a [u8]>,
    H: std::hash::BuildHasher,
{
    record_iter: I,
    batch_size: usize,
    parsers: &'a [FieldParser],
    labels: &'a ConcurrentEnumMaps,
    enum_max_values: Arc<HashMap<usize, Arc<Mutex<u32>>, H>>,
}

impl<'a, I, H> Reader<'a, I, H>
where
    I: Iterator<Item = &'a [u8]>,
    H: std::hash::BuildHasher,
{
    /// Creates a `Reader` from a byte-sequence iterator.
    pub fn new(
        record_iter: I,
        batch_size: usize,
        parsers: &'a [FieldParser],
        labels: &'a ConcurrentEnumMaps,
        enum_max_values: Arc<HashMap<usize, Arc<Mutex<u32>>, H>>,
    ) -> Self {
        Reader {
            record_iter,
            batch_size,
            parsers,
            labels,
            enum_max_values,
        }
    }

    /// Reads the next batch of records.
    ///
    /// # Errors
    ///
    /// Returns an error of parsing a field fails.
    pub fn next_batch(&mut self) -> Result<Option<record::Batch>, variable::Error> {
        let mut rows = Vec::with_capacity(self.batch_size);
        let mut csv_reader = csv_core::Reader::new();
        for _ in 0..self.batch_size {
            match self.record_iter.next() {
                Some(r) => {
                    if let Some(r) = Record::new(&mut csv_reader, r) {
                        rows.push(r)
                    }
                    // Skip invalid rows.
                }
                None => break,
            }
        }

        if rows.is_empty() {
            return Ok(None);
        }

        let mut arrays = Vec::with_capacity(self.parsers.len());
        for (i, parser) in self.parsers.iter().enumerate() {
            let col = match parser {
                FieldParser::Int64(parse) | FieldParser::Timestamp(parse) => {
                    build_primitive_array::<Int64Type, Int64Parser>(&rows, i, parse)?
                }
                FieldParser::Float64(parse) => {
                    build_primitive_array::<Float64Type, Float64Parser>(&rows, i, parse)?
                }
                FieldParser::Utf8 => {
                    let mut builder = StringBuilder::with_capacity(rows.len())?;
                    for row in &rows {
                        builder.try_push(std::str::from_utf8(row.get(i).unwrap_or_default())?)?;
                    }
                    builder.build()
                }
                FieldParser::Binary => {
                    let mut builder = BinaryBuilder::with_capacity(rows.len())?;
                    for row in &rows {
                        builder.try_push(row.get(i).unwrap_or_default())?;
                    }
                    builder.build()
                }
                FieldParser::UInt32(parse) => {
                    build_primitive_array::<UInt32Type, UInt32Parser>(&rows, i, parse)?
                }
                FieldParser::Dict => {
                    let mut builder = PrimitiveBuilder::<UInt32Type>::with_capacity(rows.len())?;
                    for r in &rows {
                        let key = std::str::from_utf8(r.get(i).unwrap_or_default())?;
                        let value = self.labels.get(&i).map_or_else(u32::max_value, |map| {
                            let mut entry = map.entry(key.to_string()).or_insert_with(|| {
                                self.enum_max_values.get(&i).map_or(
                                    (u32::max_value(), 0_usize),
                                    |v| {
                                        let mut value_locked = v.lock().expect("safe");
                                        if *value_locked < u32::max_value() {
                                            *value_locked += 1;
                                        }
                                        (*value_locked, 0_usize)
                                    },
                                )
                            });
                            *entry.value_mut() = (entry.value().0, entry.value().1 + 1);
                            entry.value().0
                            // u32::max_value means something wrong, and 0 means unmapped. And, enum value starts with 1.
                        });
                        builder.try_push(value)?;
                    }
                    builder.build()
                }
            };
            arrays.push(col);
        }
        Ok(Some(record::Batch::new(arrays)))
    }
}

pub(crate) fn build_primitive_array<T, P>(
    rows: &[Record],
    col_idx: usize,
    parse: &Arc<P>,
) -> Result<Arc<dyn Array>, AllocationError>
where
    T: PrimitiveType,
    T::Native: Default,
    P: Fn(&[u8]) -> Result<T::Native, ParseError> + Send + Sync + ?Sized,
{
    let mut builder = PrimitiveBuilder::<T>::with_capacity(rows.len())?;
    for row in rows {
        match row.get(col_idx) {
            Some(s) if !s.is_empty() => {
                let t = parse(s).unwrap_or_default();
                builder.try_push(t)?;
            }
            _ => builder.try_push(T::Native::default())?,
        }
    }
    Ok(builder.build())
}

/// Infers the data type of a field in a CSV record.
fn infer_field_type(field: &[u8]) -> DataType {
    if let Ok(s) = str::from_utf8(field) {
        if s.parse::<i64>().is_ok() {
            DataType::Int64
        } else if s.parse::<f64>().is_ok() {
            DataType::Float64
        } else {
            DataType::Utf8
        }
    } else {
        DataType::Binary
    }
}

/// Infers the schema of CSV by reading one record.
///
/// # Errors
///
/// Returns an error if there is no data to read from `reader`.
pub fn infer_schema<R: Read>(reader: &mut BufReader<R>) -> Result<Schema, String> {
    let mut csv_reader = csv_core::Reader::new();
    let record = Record::from_buf(&mut csv_reader, reader).ok_or("no data available")?;
    let mut fields = Vec::new();
    for i in 0..record.ends.len() {
        let data_type = record
            .get(i)
            .map_or(DataType::Utf8, |f| infer_field_type(f));
        fields.push(Field::new(data_type));
    }
    Ok(Schema::new(fields))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_to_datatypes() {
        let buf = "Cat,50,1.0,1990-11-28T12:00:09.0-07:00\n".as_bytes();
        let mut input = BufReader::new(buf);
        let schema = infer_schema(&mut input).unwrap();
        let answers = vec![
            Field::new(DataType::Utf8),
            Field::new(DataType::Int64),
            Field::new(DataType::Float64),
            Field::new(DataType::Utf8),
        ];

        assert!(schema
            .fields()
            .into_iter()
            .zip(answers.into_iter())
            .all(|(a, b)| a.data_type() == b.data_type()));
    }
}
