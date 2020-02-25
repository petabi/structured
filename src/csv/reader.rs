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

struct Record {
    fields: Vec<u8>,
    ends: Vec<usize>,
}

impl Record {
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

/// A parser for a single field in CSV.
#[derive(Clone)]
pub enum FieldParser {
    /// A parser converting a byte sequence into `i64`.
    Int64(Arc<Int64Parser>),

    /// A parser converting a byte sequence into `u32`.
    UInt32(Arc<UInt32Parser>),

    /// A parser converting a byte sequence into `f64`.
    Float64(Arc<Float64Parser>),

    /// A parser reading the input into a UTF-8 string.
    Utf8,

    /// A dummy parser that preserves the input byte sequence.
    Binary,

    /// A timestamp parser converting time into `i64`.
    Timestamp(Arc<Int64Parser>),

    /// A parser for nominal values, storing them using dictionary-encoding.
    Dict,
}

impl FieldParser {
    /// Creates an `i64` parser.
    #[must_use]
    pub fn int64() -> Self {
        Self::Int64(Arc::new(parse::<i64>))
    }

    /// Creates a `u32` parser.
    #[must_use]
    pub fn uint32() -> Self {
        Self::UInt32(Arc::new(parse::<u32>))
    }

    /// Creates a `f64` parser.
    #[must_use]
    pub fn float64() -> Self {
        Self::Float64(Arc::new(parse::<f64>))
    }

    /// Creates a timestamp parser that converts time into the number of
    /// non-leap seconds since the midnight on January 1, 1970.
    #[must_use]
    pub fn timestamp() -> Self {
        Self::Int64(Arc::new(parse_timestamp))
    }

    /// Creates a custom `u32` parser.
    #[must_use]
    pub fn uint32_with_parser<P>(parser: P) -> Self
    where
        P: Fn(&[u8]) -> Result<u32, ParseError> + Send + Sync + 'static,
    {
        Self::UInt32(Arc::new(parser))
    }

    /// Creates a custom timestamp parser.
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

    pub fn generate_empty_batch(&self) -> record::Batch {
        let arrays = self
            .parsers
            .iter()
            .map(|parser| match parser {
                FieldParser::Int64(_) | FieldParser::Timestamp(_) => {
                    PrimitiveBuilder::<Int64Type>::with_capacity(0)
                        .expect("fail to build empty array")
                        .build()
                }
                FieldParser::Float64(_) => PrimitiveBuilder::<Float64Type>::with_capacity(0)
                    .expect("fail to build empty array")
                    .build(),
                FieldParser::Utf8 => StringBuilder::with_capacity(0)
                    .expect("fail to build empty array")
                    .build(),
                FieldParser::Binary => BinaryBuilder::with_capacity(0)
                    .expect("fail to build empty array")
                    .build(),
                FieldParser::UInt32(_) | FieldParser::Dict => {
                    PrimitiveBuilder::<UInt32Type>::with_capacity(0)
                        .expect("fail to build empty array")
                        .build()
                }
            })
            .collect();
        record::Batch::new(arrays)
    }
}

fn build_primitive_array<T, P>(
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
    use crate::array::{Array, BinaryArray, StringArray};
    use crate::table::Column;
    use chrono::{NaiveDate, NaiveDateTime};
    use itertools::izip;
    use num_traits::ToPrimitive;
    use std::collections::HashMap;
    use std::convert::TryFrom;
    use std::net::Ipv4Addr;

    fn convert_to_conc_enum_maps(
        enum_maps: &HashMap<usize, HashMap<String, (u32, usize)>>,
    ) -> ConcurrentEnumMaps {
        let c_enum_maps = Arc::new(DashMap::new());

        for (column, map) in enum_maps {
            let c_map = Arc::new(DashMap::<String, (u32, usize)>::new());
            for (data, enum_val) in map {
                c_map.insert(data.clone(), (enum_val.0, enum_val.1));
            }
            c_enum_maps.insert(*column, c_map);
        }
        c_enum_maps
    }

    fn get_test_data() -> (
        Vec<Vec<u8>>,
        HashMap<usize, HashMap<String, (u32, usize)>>,
        Vec<Column>,
    ) {
        let c0_v: Vec<i64> = vec![1, 3, 3, 5, 2, 1, 3];
        let c1_v: Vec<_> = vec!["111a qwer", "b", "c", "d", "b", "111a qwer", "111a qwer"];
        let c2_v: Vec<Ipv4Addr> = vec![
            Ipv4Addr::new(127, 0, 0, 1),
            Ipv4Addr::new(127, 0, 0, 2),
            Ipv4Addr::new(127, 0, 0, 3),
            Ipv4Addr::new(127, 0, 0, 4),
            Ipv4Addr::new(127, 0, 0, 2),
            Ipv4Addr::new(127, 0, 0, 2),
            Ipv4Addr::new(127, 0, 0, 3),
        ];
        let c3_v: Vec<f64> = vec![2.2, 3.14, 122.8, 5.3123, 7.0, 10320.811, 5.5];
        let c4_v: Vec<NaiveDateTime> = vec![
            NaiveDate::from_ymd(2019, 9, 22).and_hms(6, 10, 11),
            NaiveDate::from_ymd(2019, 9, 22).and_hms(6, 15, 11),
            NaiveDate::from_ymd(2019, 9, 21).and_hms(20, 10, 11),
            NaiveDate::from_ymd(2019, 9, 21).and_hms(20, 10, 11),
            NaiveDate::from_ymd(2019, 9, 22).and_hms(6, 45, 11),
            NaiveDate::from_ymd(2019, 9, 21).and_hms(8, 10, 11),
            NaiveDate::from_ymd(2019, 9, 22).and_hms(9, 10, 11),
        ];
        let c5_v: Vec<u32> = vec![1, 2, 2, 2, 2, 2, 7];
        let c6_v: Vec<&[u8]> = vec![
            b"111a qwer",
            b"b",
            b"c",
            b"d",
            b"b",
            b"111a qwer",
            b"111a qwer",
        ];

        let mut c5_map: HashMap<u32, String> = HashMap::new();
        c5_map.insert(1, "t1".to_string());
        c5_map.insert(2, "t2".to_string());
        c5_map.insert(7, "t3".to_string());

        let mut data = vec![];
        let fmt = "%Y-%m-%d %H:%M:%S";
        for (c0, c1, c2, c3, c4, c5, c6) in izip!(
            c0_v.iter(),
            c1_v.iter(),
            c2_v.iter(),
            c3_v.iter(),
            c4_v.iter(),
            c5_v.iter(),
            c6_v.iter()
        ) {
            let mut row: Vec<u8> = vec![];
            row.extend(c0.to_string().into_bytes());
            row.extend_from_slice(b",");
            row.extend(c1.to_string().into_bytes());
            row.extend_from_slice(b",");
            row.extend(c2.to_string().into_bytes());
            row.extend_from_slice(b",");
            row.extend(c3.to_string().into_bytes());
            row.extend_from_slice(b",");
            row.extend(c4.format(fmt).to_string().into_bytes());
            row.extend_from_slice(b",");
            row.extend(c5_map.get(c5).unwrap().to_string().into_bytes());
            row.extend_from_slice(b",");
            row.extend_from_slice(c6);
            data.push(row);
        }

        let mut labels = HashMap::new();
        labels.insert(5, c5_map.into_iter().map(|(k, v)| (v, (k, 0))).collect());

        let c0 = Column::try_from_slice::<Int64Type>(&c0_v).unwrap();
        let c1_a: Arc<dyn Array> = Arc::new(StringArray::try_from(c1_v.as_slice()).unwrap());
        let c1 = Column::from(c1_a);
        let c2 = Column::try_from_slice::<UInt32Type>(
            c2_v.iter()
                .map(|&v| -> u32 { v.into() })
                .collect::<Vec<_>>()
                .as_slice(),
        )
        .unwrap();
        let c3 = Column::try_from_slice::<Float64Type>(&c3_v).unwrap();
        let c4 = Column::try_from_slice::<Int64Type>(
            c4_v.iter()
                .map(|v| v.timestamp())
                .collect::<Vec<_>>()
                .as_slice(),
        )
        .unwrap();
        let c5 = Column::try_from_slice::<UInt32Type>(&c5_v).unwrap();
        let c6_a: Arc<dyn Array> = Arc::new(BinaryArray::try_from(c6_v.as_slice()).unwrap());
        let c6 = Column::from(c6_a);
        let columns: Vec<Column> = vec![c0, c1, c2, c3, c4, c5, c6];
        (data, labels, columns)
    }

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

    #[test]
    fn parse_records() {
        let parsers = [
            FieldParser::int64(),
            FieldParser::Utf8,
            FieldParser::uint32_with_parser(|v| {
                let val: String = v.iter().map(|&c| c as char).collect();
                val.parse::<Ipv4Addr>().map(Into::into).map_err(Into::into)
            }),
            FieldParser::float64(),
            FieldParser::timestamp_with_parser(move |v| {
                let val: String = v.iter().map(|&c| c as char).collect();
                Ok(NaiveDateTime::parse_from_str(&val, "%Y-%m-%d %H:%M:%S")?.timestamp())
            }),
            FieldParser::Dict,
            FieldParser::Binary,
        ];
        let (data, labels, columns) = get_test_data();
        let c_enum_maps = convert_to_conc_enum_maps(&labels);
        let enum_max_values: HashMap<usize, Arc<Mutex<u32>>> = c_enum_maps
            .iter()
            .map(|m| {
                (
                    *m.key(),
                    Arc::new(Mutex::new(
                        m.value().len().to_u32().unwrap_or(u32::max_value()),
                    )),
                )
            })
            .collect();
        let mut reader = Reader::new(
            data.iter().map(|d| d.as_slice()),
            data.len(),
            &parsers,
            &c_enum_maps,
            Arc::new(enum_max_values),
        );
        let result: Vec<Column> = if let Some(batch) = reader.next_batch().unwrap() {
            batch.columns().iter().map(|c| c.clone().into()).collect()
        } else {
            Vec::new()
        };
        assert_eq!(result, columns);
    }

    #[test]
    fn missing_enum_map() {
        let parsers = [FieldParser::Dict];
        let labels = HashMap::<usize, HashMap<String, (u32, usize)>>::new();

        let record = "1\n".to_string().into_bytes();
        let row = vec![record.as_slice()];
        let c_enum_maps = convert_to_conc_enum_maps(&labels);
        let enum_max_values: HashMap<usize, Arc<Mutex<u32>>> = c_enum_maps
            .iter()
            .map(|m| {
                (
                    *m.key(),
                    Arc::new(Mutex::new(
                        m.value().len().to_u32().unwrap_or(u32::max_value()),
                    )),
                )
            })
            .collect();
        let mut reader = Reader::new(
            row.iter().cloned(),
            row.len(),
            &parsers,
            &c_enum_maps,
            Arc::new(enum_max_values),
        );
        let result: Vec<Column> = if let Some(batch) = reader.next_batch().unwrap() {
            batch.columns().iter().map(|c| c.clone().into()).collect()
        } else {
            Vec::new()
        };

        let c = Column::try_from_slice::<UInt32Type>(&[u32::max_value()][0..1]).unwrap();
        assert_eq!(c, result[0]);
    }
}
