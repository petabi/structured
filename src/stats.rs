use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::iter::Iterator;
use std::net::{IpAddr, Ipv4Addr};

use arrow::datatypes::{Float64Type, Int64Type, UInt32Type, UInt64Type};
use jiff::civil::DateTime;
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use statify::Stats;

use crate::table::{Column, ColumnType};

const MAX_TIME_INTERVAL: u32 = 86_400; // one day in seconds
const MIN_TIME_INTERVAL: u32 = 30; // seconds

/// The underlying data type of a column description.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum Element {
    Int(i64),
    UInt(u64),
    Enum(String), // describe() converts UInt -> Enum using enum maps. Without maps, by to_string().
    Float(f64),
    FloatRange(FloatRange),
    Text(String),
    Binary(Vec<u8>),
    IpAddr(IpAddr),
    DateTime(DateTime),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize, Eq, Hash)]
pub enum GroupElement {
    Int(i64),
    UInt(u32),
    Enum(String),
    Text(String),
    IpAddr(IpAddr),
    DateTime(DateTime),
}

#[derive(Debug, Default, PartialEq, Clone, Serialize, Deserialize)]
pub struct FloatRange {
    pub smallest: f64,
    pub largest: f64,
}

impl fmt::Display for Element {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(x) => write!(f, "{x}"),
            Self::UInt(x) => write!(f, "{x}"),
            Self::Enum(x) | Self::Text(x) => write!(f, "{x}"),
            Self::Binary(x) => write!(f, "{x:#?}"),
            Self::Float(x) => write!(f, "{x}"),
            Self::FloatRange(x) => {
                if x.smallest == 0.0_f64 && x.largest == 0.0_f64 {
                    write!(f, "0")
                } else {
                    write!(f, "{:.3}~{:.3}", x.smallest, x.largest)
                }
            }
            Self::IpAddr(x) => write!(f, "{x}"),
            Self::DateTime(x) => write!(f, "{x}"),
        }
    }
}

impl PartialOrd for GroupElement {
    fn partial_cmp(&self, other: &GroupElement) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Self::Int(s), Self::Int(o)) => Some(s.cmp(o)),
            (Self::UInt(s), Self::UInt(o)) => Some(s.cmp(o)),
            (Self::Enum(s), Self::Enum(o)) | (Self::Text(s), Self::Text(o)) => Some(s.cmp(o)),
            (Self::IpAddr(s), Self::IpAddr(o)) => Some(s.cmp(o)),
            (Self::DateTime(s), Self::DateTime(o)) => Some(s.cmp(o)),
            _ => None,
        }
    }
}

/// Statistical summary of data of the same type.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColumnStatistics {
    pub description: Description,
    pub n_largest_count: NLargestCount,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub struct Description {
    count: usize,
    mean: Option<f64>,
    s_deviation: Option<f64>,
    min: Option<Element>,
    max: Option<Element>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ElementCount {
    pub value: Element,
    pub count: usize,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub struct NLargestCount {
    number_of_elements: usize,
    top_n: Vec<ElementCount>,
    mode: Option<Element>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GroupElementCount {
    pub value: GroupElement,
    pub count: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GroupCount {
    pub count_index: Option<usize>, // if None, count just rows. If Some, count values of the column.
    pub series: Vec<GroupElementCount>,
}

impl fmt::Display for Description {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Start of Description")?;
        writeln!(f, "   count: {}", self.count)?;
        if self.mean.is_some() {
            writeln!(f, "   mean: {}", self.mean().unwrap())?;
        }
        if self.s_deviation.is_some() {
            writeln!(f, "   s-deviation: {}", self.std_deviation().unwrap())?;
        }
        if self.min.is_some() {
            writeln!(f, "   min: {}", self.min().unwrap())?;
        }
        if self.max.is_some() {
            writeln!(f, "   max: {}", self.max().unwrap())?;
        }
        writeln!(f, "End of Description")
    }
}

impl fmt::Display for NLargestCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Start of NLargestCount")?;
        writeln!(f, "   number of elements: {}", self.number_of_elements())?;
        writeln!(f, "   Top N")?;
        for elem in self.top_n() {
            writeln!(f, "      data: {}      count: {}", elem.value, elem.count)?;
        }
        if self.mode.is_some() {
            writeln!(f, "   mode: {}", self.mode().unwrap())?;
        }
        writeln!(f, "End of NLargestCount")
    }
}

impl Description {
    #[must_use]
    pub fn new(
        count: usize,
        mean: Option<f64>,
        s_deviation: Option<f64>,
        min: Option<Element>,
        max: Option<Element>,
    ) -> Self {
        Self {
            count,
            mean,
            s_deviation,
            min,
            max,
        }
    }

    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }

    #[must_use]
    pub fn mean(&self) -> Option<f64> {
        self.mean
    }

    #[must_use]
    pub fn std_deviation(&self) -> Option<f64> {
        self.s_deviation
    }

    #[must_use]
    pub fn min(&self) -> Option<&Element> {
        self.min.as_ref()
    }

    #[must_use]
    pub fn max(&self) -> Option<&Element> {
        self.max.as_ref()
    }
}

impl NLargestCount {
    #[must_use]
    pub fn new(number_of_elements: usize, top_n: Vec<ElementCount>, mode: Option<Element>) -> Self {
        Self {
            number_of_elements,
            top_n,
            mode,
        }
    }

    #[must_use]
    pub fn number_of_elements(&self) -> usize {
        self.number_of_elements
    }

    #[must_use]
    pub fn top_n(&self) -> &Vec<ElementCount> {
        &self.top_n
    }

    #[must_use]
    pub fn mode(&self) -> Option<&Element> {
        self.mode.as_ref()
    }
}

macro_rules! min_max {
    ( $iter:expr, $d:expr, $t2:expr ) => {{
        if let Some(minmax) = find_min_max($iter) {
            $d.min = Some($t2(minmax.min));
            $d.max = Some($t2(minmax.max));
        } else {
            $d.min = None;
            $d.max = None;
        }
    }};
}

fn set_mean_deviation(values: &Vec<f64>, description: &mut Description) {
    let mut mean = values
        .mean()
        .unwrap_or_else(|_| panic!("population variance requires at least one data point"));
    // The previous sum started at +0.0, so a column of negative zeros had a +0.0 mean.
    if mean.to_bits() == (-0.0_f64).to_bits()
        && values
            .iter()
            .all(|value| value.to_bits() == (-0.0_f64).to_bits())
    {
        mean = 0.0;
    }
    let deviation = values.std_dev_pop().expect("validated nonempty sample");
    assert!(deviation >= 0.0, "negative sum of square root deviations");
    description.mean = Some(mean);
    description.s_deviation = Some(deviation);
}

macro_rules! top_n {
    ( $iter:expr, $len:expr, $d:expr, $t1:ty, $t2:expr, $num_of_top_n:expr ) => {
        let top_n_native: Vec<($t1, usize)> = count_sort($iter);
        $d.number_of_elements = top_n_native.len();
        let mut top_n: Vec<ElementCount> = Vec::new();
        let num_of_top_n = $num_of_top_n.to_usize().expect("safe: u32 -> usize");
        let top_n_num = if num_of_top_n > top_n_native.len() {
            top_n_native.len()
        } else {
            num_of_top_n
        };
        for (x, y) in &top_n_native[0..top_n_num] {
            top_n.push(ElementCount {
                value: $t2((*x).to_owned()),
                count: *y,
            });
        }
        $d.mode = top_n.first().map(|v| v.value.clone());
        $d.top_n = top_n;
    };
}

#[must_use]
pub(crate) fn describe(column: &Column, rows: &[usize], column_type: ColumnType) -> Description {
    let mut description = Description {
        count: rows.len(),
        ..Description::default()
    };

    match column_type {
        ColumnType::Int64 => {
            let iter = column.primitive_iter::<Int64Type>(rows).unwrap();
            min_max!(iter, description, Element::Int);
            let iter = column.primitive_iter::<Int64Type>(rows).unwrap();
            #[allow(clippy::cast_precision_loss)] // 52-bit precision is good enough
            let f_values: Vec<f64> = iter.map(|v: i64| v as f64).collect();
            set_mean_deviation(&f_values, &mut description);
        }
        ColumnType::Float64 => {
            let iter = column.primitive_iter::<Float64Type>(rows).unwrap();
            min_max!(iter, description, Element::Float);
            let iter = column.primitive_iter::<Float64Type>(rows).unwrap();
            let values = iter.collect::<Vec<_>>();
            set_mean_deviation(&values, &mut description);
        }
        _ => (),
    }

    description
}

#[must_use]
pub(crate) fn n_largest_count(
    column: &Column,
    rows: &[usize],
    column_type: ColumnType,
    number_of_top_n: u32,
) -> NLargestCount {
    let mut n_largest_count = NLargestCount::default();

    match column_type {
        ColumnType::Int64 => {
            let iter = column.primitive_iter::<Int64Type>(rows).unwrap();
            top_n!(
                iter,
                rows.len(),
                n_largest_count,
                i64,
                Element::Int,
                number_of_top_n
            );
        }
        ColumnType::Enum => {
            let iter = column.primitive_iter::<UInt64Type>(rows).unwrap();
            top_n!(
                iter,
                rows.len(),
                n_largest_count,
                u64,
                Element::UInt,
                number_of_top_n
            );
        }
        ColumnType::Utf8 => {
            let iter = column.string_iter(rows).unwrap();
            top_n!(
                iter,
                rows.len(),
                n_largest_count,
                &str,
                Element::Text,
                number_of_top_n
            );
        }
        ColumnType::Binary => {
            let iter = column.binary_iter(rows).unwrap();
            top_n!(
                iter,
                rows.len(),
                n_largest_count,
                &[u8],
                Element::Binary,
                number_of_top_n
            );
        }
        ColumnType::IpAddr => {
            let values = column
                .primitive_iter::<UInt32Type>(rows)
                .unwrap()
                .map(|v| IpAddr::from(Ipv4Addr::from(v)))
                .collect::<Vec<_>>();
            top_n!(
                values.iter(),
                rows.len(),
                n_largest_count,
                &IpAddr,
                Element::IpAddr,
                number_of_top_n
            );
        }
        ColumnType::DateTime | ColumnType::Float64 => unreachable!(), // by implementation
    }

    n_largest_count
}

#[must_use]
#[allow(clippy::too_many_lines)]
pub(crate) fn n_largest_count_enum(
    column: &Column,
    rows: &[usize],
    reverse_map: &HashMap<u64, Vec<String>>,
    number_of_top_n: u32,
) -> NLargestCount {
    let n_largest_count = n_largest_count(column, rows, ColumnType::Enum, number_of_top_n);

    let (top_n, mode) = {
        if reverse_map.is_empty() {
            (
                n_largest_count
                    .top_n()
                    .iter()
                    .map(|elem| {
                        if let Element::UInt(value) = elem.value {
                            ElementCount {
                                value: Element::Enum(value.to_string()),
                                count: elem.count,
                            }
                        } else {
                            ElementCount {
                                value: Element::Enum("_N/A_".to_string()),
                                count: elem.count,
                            }
                        }
                    })
                    .collect(),
                match n_largest_count.mode() {
                    Some(mode) => {
                        if let Element::UInt(value) = mode {
                            Some(Element::Enum(value.to_string()))
                        } else {
                            None
                        }
                    }
                    None => None,
                },
            )
        } else {
            (
                n_largest_count
                    .top_n()
                    .iter()
                    .map(|elem| {
                        if let Element::UInt(value) = elem.value {
                            ElementCount {
                                value: Element::Enum(reverse_map.get(&value).map_or(
                                    "_NO_MAP_".to_string(),
                                    |v| {
                                        let mut s = String::new();
                                        for (i, e) in v.iter().enumerate() {
                                            s.push_str(e);
                                            if i < v.len() - 1 {
                                                s.push('|');
                                            }
                                        }
                                        s
                                    },
                                )),
                                count: elem.count,
                            }
                        } else {
                            ElementCount {
                                value: Element::Enum("_N/A_".to_string()),
                                count: elem.count,
                            }
                        }
                    })
                    .collect(),
                match n_largest_count.mode() {
                    Some(mode) => {
                        if let Element::UInt(value) = mode {
                            Some(Element::Enum(reverse_map.get(value).map_or(
                                "_NO_MAP_".to_string(),
                                |v| {
                                    let mut s = String::new();
                                    for (i, e) in v.iter().enumerate() {
                                        s.push_str(e);
                                        if i < v.len() - 1 {
                                            s.push('|');
                                        }
                                    }
                                    s
                                },
                            )))
                        } else {
                            None
                        }
                    }
                    None => None,
                },
            )
        }
    };

    NLargestCount {
        number_of_elements: n_largest_count.number_of_elements(),
        top_n,
        mode,
    }
}

#[must_use]
pub(crate) fn n_largest_count_float64(
    column: &Column,
    rows: &[usize],
    number_of_top_n: u32,
    precision: i32,
) -> NLargestCount {
    let mut n_largest_count = NLargestCount::default();

    let iter = column.primitive_iter::<Float64Type>(rows).unwrap();
    let (rc, rt) = top_n_f64(iter, 10.0_f64.powi(precision), number_of_top_n);
    n_largest_count.number_of_elements = rc;
    n_largest_count.mode = Some(rt[0].value.clone());
    n_largest_count.top_n = rt;

    n_largest_count
}

#[must_use]
pub(crate) fn n_largest_count_datetime(
    column: &Column,
    rows: &[usize],
    time_interval: u32,
    number_of_top_n: u32,
) -> NLargestCount {
    let mut n_largest_count = NLargestCount::default();
    let values = convert_time_intervals(column, rows, time_interval);

    top_n!(
        values.iter(),
        rows.len(),
        n_largest_count,
        &DateTime,
        Element::DateTime,
        number_of_top_n
    );

    n_largest_count
}

/// # Panics
///
/// If `rows` contains an invalid timestamp in nanoseconds.
#[must_use]
pub(crate) fn convert_time_intervals(
    column: &Column,
    rows: &[usize],
    time_interval: u32,
) -> Vec<DateTime> {
    const A_BILLION: i64 = 1_000_000_000;
    let time_interval = if time_interval > MAX_TIME_INTERVAL {
        MAX_TIME_INTERVAL
    // Users want to see time series of the same order intervals within MAX_TIME_INTERVAL which is in date units.
    // If the interval is larger than a day, it should be in date units.
    } else {
        time_interval
    };
    let time_interval = if time_interval < MIN_TIME_INTERVAL {
        MIN_TIME_INTERVAL
    } else {
        time_interval
    };
    let time_interval = i64::from(time_interval);

    column
        .primitive_iter::<Int64Type>(rows)
        .unwrap()
        .map(|v| {
            // The first interval of each day should start with 00:00:00.
            let mut interval_idx = v / A_BILLION;
            interval_idx = (interval_idx / time_interval) * time_interval;
            jiff::Timestamp::from_second(interval_idx)
                .unwrap_or_default()
                .to_zoned(jiff::tz::TimeZone::UTC)
                .datetime()
        })
        .collect::<Vec<_>>()
}

fn count_sort<I>(iter: I) -> Vec<(I::Item, usize)>
where
    I: Iterator,
    I::Item: Clone + Eq + Hash,
{
    let mut count: HashMap<I::Item, usize> = HashMap::new();
    for v in iter {
        let c = count.entry(v).or_insert(0);
        *c += 1;
    }
    let mut top_n: Vec<(I::Item, usize)> = Vec::new();
    for (k, v) in &count {
        top_n.push(((*k).clone(), *v));
    }
    top_n.sort_unstable_by_key(|a| std::cmp::Reverse(a.1));
    top_n
}

fn top_n_f64<I>(iter: I, precision: f64, number_of_top_n: u32) -> (usize, Vec<ElementCount>)
where
    I: Iterator<Item = f64>,
{
    use ordered_float::OrderedFloat;

    let mut freqs: Vec<(_, usize)> = iter
        .map(|v| OrderedFloat((v * precision).round() / precision))
        .fold(HashMap::new(), |mut freqs, v| {
            let e = freqs.entry(v).or_default();
            *e += 1;
            freqs
        })
        .into_iter()
        .collect();

    freqs.sort_unstable_by_key(|a| std::cmp::Reverse(a.1));

    (
        freqs.len(),
        freqs
            .into_iter()
            .take(number_of_top_n.to_usize().expect("safe: u32 -> usize"))
            .map(|(v, count)| ElementCount {
                value: Element::Float(v.into_inner()),
                count,
            })
            .collect(),
    )
}

struct MinMax<T> {
    min: T,
    max: T,
}

/// Returns the minimum and maximum values.
fn find_min_max<I>(mut iter: I) -> Option<MinMax<I::Item>>
where
    I: Iterator,
    I::Item: Copy + PartialOrd,
{
    let mut min = iter.next()?;
    let mut max = min;

    for v in iter {
        if min > v {
            min = v;
        } else if max < v {
            max = v;
        }
    }
    Some(MinMax { min, max })
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::Int64Type;
    use jiff::civil::date;

    use super::*;
    use crate::Column;

    #[test]
    fn describe_integer_column_reports_population_statistics() {
        let column = Column::try_from_slice::<Int64Type>(&[1, 2, 3]).unwrap();
        let description = describe(&column, &[0, 1, 2], ColumnType::Int64);
        assert_eq!(description.mean(), Some(2.0));
        assert_eq!(description.std_deviation(), Some((2.0_f64 / 3.0).sqrt()));
    }

    #[test]
    fn describe_float_column_preserves_statistics() {
        let cases: &[(&[f64], f64, f64)] = &[
            (&[1.0, 2.0, 3.0], 2.0, (2.0_f64 / 3.0).sqrt()),
            (&[1e155], 1e155, 0.0),
            (
                &[1e16, 1e16 + 2.0, 1e16 + 4.0],
                1e16 + 2.0,
                (8.0_f64 / 3.0).sqrt(),
            ),
            (&[-0.0, -0.0], 0.0, 0.0),
            (&[-f64::from_bits(1), 0.0], -0.0, 0.0),
        ];
        for &(values, mean, deviation) in cases {
            let column = Column::try_from_slice::<Float64Type>(values).unwrap();
            let rows: Vec<usize> = (0..values.len()).collect();
            let description = describe(&column, &rows, ColumnType::Float64);
            assert_eq!(description.mean().unwrap().to_bits(), mean.to_bits());
            assert_eq!(
                description.std_deviation().unwrap().to_bits(),
                deviation.to_bits()
            );
        }
    }

    #[test]
    #[should_panic(expected = "population variance requires at least one data point")]
    fn describe_float_column_panics_for_empty_selection() {
        let column = Column::try_from_slice::<Float64Type>(&[1.0]).unwrap();
        let _ = describe(&column, &[], ColumnType::Float64);
    }

    #[test]
    fn describe_float_column_panics_for_nonfinite_values() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let result = std::panic::catch_unwind(|| {
                let column = Column::try_from_slice::<Float64Type>(&[value]).unwrap();
                describe(&column, &[0], ColumnType::Float64)
            });
            assert!(result.is_err(), "expected a panic for {value}");
        }
    }

    #[test]
    fn test_convert_time_intervals() {
        let c4_v: Vec<i64> = vec![
            date(2019, 9, 22)
                .at(6, 10, 11, 0)
                .to_zoned(jiff::tz::TimeZone::UTC)
                .unwrap()
                .timestamp()
                .as_nanosecond()
                .try_into()
                .unwrap(),
            date(2019, 9, 22)
                .at(6, 15, 11, 0)
                .to_zoned(jiff::tz::TimeZone::UTC)
                .unwrap()
                .timestamp()
                .as_nanosecond()
                .try_into()
                .unwrap(),
            date(2019, 9, 21)
                .at(20, 10, 11, 0)
                .to_zoned(jiff::tz::TimeZone::UTC)
                .unwrap()
                .timestamp()
                .as_nanosecond()
                .try_into()
                .unwrap(),
            date(2019, 9, 21)
                .at(20, 10, 11, 0)
                .to_zoned(jiff::tz::TimeZone::UTC)
                .unwrap()
                .timestamp()
                .as_nanosecond()
                .try_into()
                .unwrap(),
            date(2019, 9, 22)
                .at(6, 45, 11, 0)
                .to_zoned(jiff::tz::TimeZone::UTC)
                .unwrap()
                .timestamp()
                .as_nanosecond()
                .try_into()
                .unwrap(),
            date(2019, 9, 21)
                .at(8, 10, 11, 0)
                .to_zoned(jiff::tz::TimeZone::UTC)
                .unwrap()
                .timestamp()
                .as_nanosecond()
                .try_into()
                .unwrap(),
            date(2019, 9, 22)
                .at(9, 10, 11, 0)
                .to_zoned(jiff::tz::TimeZone::UTC)
                .unwrap()
                .timestamp()
                .as_nanosecond()
                .try_into()
                .unwrap(),
        ];
        let c4 = Column::try_from_slice::<Int64Type>(&c4_v).unwrap();
        let rows = vec![0_usize, 3, 1, 4, 2, 6, 5];
        let time_interval = 3600;
        let rst = convert_time_intervals(&c4, &rows, time_interval);
        assert_eq!(rst.len(), 7);
        assert_eq!(rst.first(), Some(&date(2019, 9, 22).at(6, 0, 0, 0)));
        assert_eq!(rst.last(), Some(&date(2019, 9, 21).at(8, 0, 0, 0)));
    }

    #[test]
    fn test_the_first_interval_of_each_day() {
        let c4_v: Vec<i64> = vec![
            date(2019, 9, 22)
                .at(0, 3, 20, 0)
                .to_zoned(jiff::tz::TimeZone::UTC)
                .unwrap()
                .timestamp()
                .as_nanosecond()
                .try_into()
                .unwrap(),
            date(2019, 9, 22)
                .at(0, 9, 11, 0)
                .to_zoned(jiff::tz::TimeZone::UTC)
                .unwrap()
                .timestamp()
                .as_nanosecond()
                .try_into()
                .unwrap(),
            date(2019, 9, 22)
                .at(0, 10, 11, 0)
                .to_zoned(jiff::tz::TimeZone::UTC)
                .unwrap()
                .timestamp()
                .as_nanosecond()
                .try_into()
                .unwrap(),
            date(2019, 9, 22)
                .at(1, 15, 11, 0)
                .to_zoned(jiff::tz::TimeZone::UTC)
                .unwrap()
                .timestamp()
                .as_nanosecond()
                .try_into()
                .unwrap(),
        ];
        let c4 = Column::try_from_slice::<Int64Type>(&c4_v).unwrap();
        let rows = vec![0_usize, 1, 2, 3];
        let time_interval = 3600;
        let rst = convert_time_intervals(&c4, &rows, time_interval);
        let converted = [
            date(2019, 9, 22).at(0, 0, 0, 0),
            date(2019, 9, 22).at(0, 0, 0, 0),
            date(2019, 9, 22).at(0, 0, 0, 0),
            date(2019, 9, 22).at(1, 0, 0, 0),
        ];
        for (seq, c) in converted.iter().enumerate() {
            assert_eq!(rst.get(seq), Some(c));
        }

        let time_interval = 600;
        let rst = convert_time_intervals(&c4, &rows, time_interval);
        let converted = [
            date(2019, 9, 22).at(0, 0, 0, 0),
            date(2019, 9, 22).at(0, 0, 0, 0),
            date(2019, 9, 22).at(0, 10, 0, 0),
            date(2019, 9, 22).at(1, 10, 0, 0),
        ];
        for (seq, c) in converted.iter().enumerate() {
            assert_eq!(rst.get(seq), Some(c));
        }
    }
}
