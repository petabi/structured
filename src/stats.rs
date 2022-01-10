use arrow::datatypes::{Float64Type, Int64Type, UInt32Type, UInt64Type};
use chrono::NaiveDateTime;
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use statistical::{mean, population_standard_deviation};
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::iter::Iterator;
use std::net::{IpAddr, Ipv4Addr};
use std::vec;

use crate::table::{Column, ColumnType};

const NUM_OF_FLOAT_INTERVALS: usize = 100;
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
    DateTime(NaiveDateTime),
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize, Eq, Hash)]
pub enum GroupElement {
    Int(i64),
    UInt(u32),
    Enum(String),
    Text(String),
    IpAddr(IpAddr),
    DateTime(NaiveDateTime),
}

#[derive(Debug, Default, PartialEq, Clone, Serialize, Deserialize)]
pub struct FloatRange {
    pub smallest: f64,
    pub largest: f64,
}

impl fmt::Display for Element {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(x) => write!(f, "{}", x),
            Self::UInt(x) => write!(f, "{}", x),
            Self::Enum(x) | Self::Text(x) => write!(f, "{}", x),
            Self::Binary(x) => write!(f, "{:#?}", x),
            Self::Float(x) => write!(f, "{}", x),
            Self::FloatRange(x) => {
                if x.smallest == 0.0_f64 && x.largest == 0.0_f64 {
                    write!(f, "0")
                } else {
                    write!(f, "{:.3}~{:.3}", x.smallest, x.largest)
                }
            }
            Self::IpAddr(x) => write!(f, "{}", x),
            Self::DateTime(x) => write!(f, "{}", x),
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GroupElementCount {
    pub value: GroupElement,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

macro_rules! mean_deviation {
    ( $vf:expr, $t1:ty, $d:expr ) => {
        let m = mean(&$vf);
        $d.mean = Some(m);
        $d.s_deviation = Some(population_standard_deviation(&$vf, Some(m)));
    };
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
            mean_deviation!(f_values, i64, description);
        }
        ColumnType::Float64 => {
            let iter = column.primitive_iter::<Float64Type>(rows).unwrap();
            min_max!(iter, description, Element::Float);
            let iter = column.primitive_iter::<Float64Type>(rows).unwrap();
            let values = iter.collect::<Vec<_>>();
            mean_deviation!(values, f64, description);
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
    min: f64,
    max: f64,
) -> NLargestCount {
    let mut n_largest_count = NLargestCount::default();

    let iter = column.primitive_iter::<Float64Type>(rows).unwrap();
    let (rc, rt) = top_n_f64(iter, min, max, number_of_top_n);
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
        &NaiveDateTime,
        Element::DateTime,
        number_of_top_n
    );

    n_largest_count
}

#[must_use]
pub(crate) fn convert_time_intervals(
    column: &Column,
    rows: &[usize],
    time_interval: u32,
) -> Vec<NaiveDateTime> {
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
            let mut ts = v / (24 * 60 * 60) * (24 * 60 * 60);
            ts += (v - ts) / time_interval * time_interval;
            NaiveDateTime::from_timestamp(ts, 0)
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
    top_n.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    top_n
}

fn top_n_f64<I>(iter: I, min: f64, max: f64, number_of_top_n: u32) -> (usize, Vec<ElementCount>)
where
    I: Iterator<Item = f64>,
{
    let interval: f64 = (max - min) / NUM_OF_FLOAT_INTERVALS.to_f64().expect("<= 100");
    let mut count: Vec<(usize, usize)> = vec![(0, 0); NUM_OF_FLOAT_INTERVALS];

    for (i, item) in count.iter_mut().enumerate().take(NUM_OF_FLOAT_INTERVALS) {
        item.0 = i;
    }

    for v in iter {
        let mut slot = if interval == 0.0_f64 {
            0_usize
        } else {
            ((v - min) / interval).floor().to_usize().expect("< 100")
        };
        if slot == NUM_OF_FLOAT_INTERVALS {
            slot = NUM_OF_FLOAT_INTERVALS - 1;
        }
        count[slot].1 += 1;
    }

    count.retain(|&c| c.1 > 0);

    count.sort_unstable_by(|a, b| b.1.cmp(&a.1));

    let mut top_n: Vec<ElementCount> = Vec::new();

    let number_of_top_n = number_of_top_n.to_usize().expect("safe: u32 -> usize");
    let num_top_n = if number_of_top_n > count.len() {
        count.len()
    } else {
        number_of_top_n
    };

    for item in count.iter().take(num_top_n) {
        if item.1 == 0 {
            break;
        }
        top_n.push(ElementCount {
            value: Element::FloatRange(FloatRange {
                smallest: min + (item.0).to_f64().expect("< 30") * interval,
                largest: min + (item.0 + 1).to_f64().expect("<= 30") * interval,
            }),
            count: item.1,
        });
    }
    (count.len(), top_n)
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
