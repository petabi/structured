use chrono::{NaiveDateTime, NaiveTime, Timelike};
use dashmap::DashMap;
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use statistical::*;
use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::fmt;
use std::hash::Hash;
use std::net::{IpAddr, Ipv4Addr};
use std::slice::Iter;
use std::sync::Arc;

use crate::{DataType, Schema};

const NUM_OF_FLOAT_INTERVALS: usize = 100;
const NUM_OF_TOP_N: usize = 30;

type ConcurrentEnumMaps = Arc<DashMap<usize, Arc<DashMap<String, (u32, usize)>>>>;
type ReverseEnumMaps = Arc<HashMap<usize, Arc<HashMap<u32, Vec<String>>>>>;

#[derive(Clone, Copy, Debug)] // same as remake::csv::ColumnType
pub enum ColumnType {
    Int64,
    Float64,
    DateTime,
    IpAddr,
    Enum,
    Utf8,
}

#[derive(Debug, Default, Clone)]
pub struct Table {
    columns: Vec<Column>,
    event_ids: HashMap<u64, usize>,
}

impl Table {
    pub fn new(columns: Vec<Column>, event_ids: HashMap<u64, usize>) -> Self {
        Self { columns, event_ids }
    }

    pub fn from_schema(schema: &Schema) -> Self {
        let columns = schema
            .fields()
            .iter()
            .map(|field| match field.data_type() {
                DataType::Int64 | DataType::Timestamp(_) => Column::new::<i64>(),
                DataType::UInt8 => Column::new::<u8>(),
                DataType::UInt32 => Column::new::<u32>(),
                DataType::Float64 => Column::new::<f64>(),
                DataType::Utf8 => Column::new::<String>(),
            })
            .collect();
        Self {
            columns,
            event_ids: HashMap::new(),
        }
    }

    /// Moves all the rows of `other` intot `self`, leaving `other` empty.
    ///
    /// # Panics
    ///
    /// Panics if the types of columns are different, or the number of rows
    /// overflows `usize`.
    pub fn append(&mut self, other: &mut Self) {
        for (self_col, other_col) in self.columns.iter_mut().zip(other.columns.iter_mut()) {
            self_col.append(other_col);
        }
    }

    /// Returns an `Iterator` for columns.
    pub fn columns(&self) -> Iter<Column> {
        self.columns.iter()
    }

    /// Returns a `Column` for the given column index.
    #[allow(dead_code)] // Used by tests only.
    pub fn get_column<T: 'static>(&self, index: usize) -> Option<&ColumnData<T>> {
        let col = self.columns.get(index)?;
        col.values()
    }

    /// Returns the number of columns in the table.
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Returns the number of rows in the table.
    pub fn num_rows(&self) -> usize {
        if self.columns.is_empty() {
            0_usize
        } else {
            let col = &self.columns[0];
            col.len()
        }
    }

    pub fn describe(
        &self,
        rows: &[usize],
        column_types: &Arc<Vec<ColumnType>>,
        r_enum_maps: &ReverseEnumMaps,
    ) -> Vec<Description> {
        self.columns
            .iter()
            .enumerate()
            .map(|(index, column)| {
                if let ColumnType::Enum = column_types[index] {
                    column.describe_enum(
                        rows,
                        r_enum_maps.get(&index).unwrap_or(&Arc::new(HashMap::new())),
                    )
                } else {
                    column.describe(rows, column_types[index])
                }
            })
            .collect()
    }

    pub fn get_index_of_event(&self, eventid: u64) -> Option<&usize> {
        self.event_ids.get(&eventid)
    }

    pub fn limit_dimension(
        &mut self,
        enum_dimensions: &HashMap<usize, u32>,
        enum_maps: &ConcurrentEnumMaps,
        max_dimension: u32,
        max_enum_portion: f64,
    ) -> (HashMap<usize, u32>, HashMap<usize, u32>) {
        let mut enum_portion_dimensions = HashMap::<usize, u32>::new();
        let mut enum_set_dimensions = HashMap::<usize, u32>::new();
        for map in enum_maps.iter() {
            let (column_index, column_map) = (map.key(), map.value());
            let dimension = (*(enum_dimensions.get(column_index).unwrap_or(&max_dimension)))
                .to_usize()
                .expect("safe");

            let mut number_of_events = 0_usize;
            let mut map_vector: Vec<(String, u32, usize)> = column_map
                .iter()
                .map(|m| {
                    number_of_events += m.value().1;
                    (m.key().clone(), m.value().0, m.value().1)
                })
                .collect();
            map_vector.sort_unstable_by(|a, b| b.2.cmp(&a.2));
            let max_of_events = (number_of_events.to_f64().expect("safe") * max_enum_portion)
                .to_usize()
                .expect("safe");
            let mut count_of_events = 0_usize;
            let mut index = 0_usize;
            for (i, m) in map_vector.iter().enumerate() {
                index = i;
                count_of_events += m.2;
                if count_of_events > max_of_events {
                    break;
                }
            }

            let truncate_dimension = if index + 1 < dimension - 1 {
                index + 1
            } else if dimension > 0 {
                dimension - 1
            } else {
                0
            };
            map_vector.truncate(truncate_dimension);

            enum_portion_dimensions.insert(*column_index, (index + 1).to_u32().expect("safe"));
            enum_set_dimensions.insert(
                *column_index,
                (truncate_dimension + 1).to_u32().expect("safe"),
            );

            let mapped_enums = map_vector.iter().map(|v| v.1).collect();
            self.limit_enum_values(*column_index, &mapped_enums);
        }
        (enum_portion_dimensions, enum_set_dimensions)
    }

    fn limit_enum_values(&mut self, column_index: usize, mapped_enums: &HashSet<u32>) {
        let cd: &mut ColumnData<u32> = self.columns[column_index].values_mut().unwrap();
        for value in cd {
            if !mapped_enums.contains(value) {
                *value = 0_u32; // if unmapped out of the predefined rate, enum value set to 0_u32.
            }
        }
    }
}

impl TryFrom<Vec<Column>> for Table {
    type Error = &'static str;

    fn try_from(columns: Vec<Column>) -> Result<Self, Self::Error> {
        let len = if let Some(col) = columns.first() {
            col.len()
        } else {
            return Ok(Self {
                columns,
                event_ids: HashMap::new(),
            });
        };
        if columns.iter().skip(1).all(|e| e.len() == len) {
            Ok(Self {
                columns,
                event_ids: HashMap::new(),
            })
        } else {
            Err("columns must have the same length")
        }
    }
}

macro_rules! column_len {
    ( $s:expr, $t:ty ) => {
        let col = $s
            .inner
            .downcast_ref::<ColumnData<$t>>()
            .expect("column type mismatch");
        return col.len();
    };
}

macro_rules! column_append {
    ( $dst:expr, $src:expr, $t:ty ) => {
        let dst_col = $dst
            .downcast_mut::<ColumnData<$t>>()
            .expect("column type mismatch");
        let src_col = $src
            .downcast_mut::<ColumnData<$t>>()
            .expect("column type mismatch");
        dst_col.append(src_col);
    };
}

macro_rules! describe_all {
    ( $rows:expr, $cd:expr, $d:expr, $t1:ty, $t2:expr) => {
        describe_min_max!($rows, $cd, $d, $t2);
        describe_top_n!($rows, $cd, $d, $t2);
        describe_mean_deviation!($rows, $cd, $d);
    };
}

macro_rules! describe_min_max {
    ( $rows:expr, $cd:expr, $d:expr, $t2:expr ) => {{
        let (min, max) = find_min_max($rows, $cd);
        $d.min = Some($t2(min));
        $d.max = Some($t2(max));
    }};
}

macro_rules! describe_mean_deviation {
    ( $rows:expr, $cd:expr, $d:expr ) => {
        let vf: Vec<f64> = $rows.iter().map(|r| $cd[*r] as f64).collect();
        let m = mean(&vf);
        $d.mean = Some(m);
        $d.s_deviation = Some(population_standard_deviation(&vf, Some(m)));
    };
}

macro_rules! describe_top_n {
    ( $rows:expr, $cd:expr, $d:expr, $t2:expr ) => {
        let top_n_native = count_sort($rows, $cd);
        $d.count = $rows.len();
        $d.unique_count = top_n_native.len();
        let mut top_n: Vec<(DescriptionElement, usize)> = Vec::new();
        let top_n_num = if NUM_OF_TOP_N > top_n_native.len() {
            top_n_native.len()
        } else {
            NUM_OF_TOP_N
        };
        for (x, y) in &top_n_native[0..top_n_num] {
            top_n.push(($t2(x.clone()), *y));
        }
        $d.mode = Some(top_n[0].0.clone());
        $d.top_n = Some(top_n);
    };
}

pub struct Column {
    inner: Box<dyn Any + Send + Sync>,
}

type ColumnData<T> = Vec<T>;

impl Column {
    pub fn new<T>() -> Self
    where
        T: Send + Sync + 'static,
    {
        Self {
            inner: Box::new(ColumnData::<T>::new()),
        }
    }

    pub fn with_data<T>(data: ColumnData<T>) -> Self
    where
        T: Send + Sync + 'static,
    {
        Self {
            inner: Box::new(data),
        }
    }

    fn len(&self) -> usize {
        if self.inner.is::<ColumnData<i64>>() {
            column_len!(self, i64);
        } else if self.inner.is::<ColumnData<f64>>() {
            column_len!(self, f64);
        } else if self.inner.is::<ColumnData<u32>>() {
            column_len!(self, u32);
        } else if self.inner.is::<ColumnData<String>>() {
            column_len!(self, String);
        } else if self.inner.is::<ColumnData<NaiveDateTime>>() {
            column_len!(self, NaiveDateTime);
        } else {
            panic!("invalid column type")
        }
    }

    fn append(&mut self, other: &mut Self) {
        if self.inner.is::<ColumnData<i64>>() {
            column_append!(self.inner, other.inner, i64);
        } else if self.inner.is::<ColumnData<f64>>() {
            column_append!(self.inner, other.inner, f64);
        } else if self.inner.is::<ColumnData<u32>>() {
            column_append!(self.inner, other.inner, u32);
        } else if self.inner.is::<ColumnData<String>>() {
            column_append!(self.inner, other.inner, String);
        } else if self.inner.is::<ColumnData<NaiveDateTime>>() {
            column_append!(self.inner, other.inner, NaiveDateTime);
        } else {
            panic!("invalid column type");
        }
    }

    /// Returns the data if the type matches.
    pub fn values<T: 'static>(&self) -> Option<&ColumnData<T>> {
        self.inner.downcast_ref::<ColumnData<T>>()
    }

    /// Returns the mutable data if the type matches.
    pub fn values_mut<T: 'static>(&mut self) -> Option<&mut ColumnData<T>> {
        self.inner.downcast_mut::<ColumnData<T>>()
    }

    pub fn describe_enum(
        &self,
        rows: &[usize],
        reverse_map: &Arc<HashMap<u32, Vec<String>>>,
    ) -> Description {
        let desc = self.describe(rows, ColumnType::Enum);

        let (top_n, mode) = {
            if reverse_map.is_empty() {
                (
                    match desc.get_top_n() {
                        Some(top_n) => Some(
                            top_n
                                .iter()
                                .map(|(v, c)| {
                                    if let DescriptionElement::UInt(value) = v {
                                        (DescriptionElement::Enum(value.to_string()), *c)
                                    } else {
                                        (DescriptionElement::Enum("_N/A_".to_string()), *c)
                                    }
                                })
                                .collect(),
                        ),
                        None => None,
                    },
                    match desc.get_mode() {
                        Some(mode) => {
                            if let DescriptionElement::UInt(value) = mode {
                                Some(DescriptionElement::Enum(value.to_string()))
                            } else {
                                None
                            }
                        }
                        None => None,
                    },
                )
            } else {
                (
                    match desc.get_top_n() {
                        Some(top_n) => Some(
                            top_n
                                .iter()
                                .map(|(v, c)| {
                                    if let DescriptionElement::UInt(value) = v {
                                        (
                                            DescriptionElement::Enum(
                                                reverse_map.get(value).map_or(
                                                    "_NO_MAP_".to_string(),
                                                    |v| {
                                                        let mut s = String::new();
                                                        for (i, e) in v.iter().enumerate() {
                                                            s.push_str(e);
                                                            if i < v.len() - 1 {
                                                                s.push_str("|")
                                                            }
                                                        }
                                                        s
                                                    },
                                                ),
                                            ),
                                            *c,
                                        )
                                    } else {
                                        (DescriptionElement::Enum("_N/A_".to_string()), *c)
                                    }
                                })
                                .collect(),
                        ),
                        None => None,
                    },
                    match desc.get_mode() {
                        Some(mode) => {
                            if let DescriptionElement::UInt(value) = mode {
                                Some(DescriptionElement::Enum(reverse_map.get(value).map_or(
                                    "_NO_MAP_".to_string(),
                                    |v| {
                                        let mut s = String::new();
                                        for (i, e) in v.iter().enumerate() {
                                            s.push_str(e);
                                            if i < v.len() - 1 {
                                                s.push_str("|")
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

        Description {
            count: desc.get_count(),
            unique_count: desc.get_unique_count(),
            mean: None,
            s_deviation: None,
            min: None,
            max: None,
            top_n,
            mode,
        }
    }

    pub fn describe(&self, rows: &[usize], column_type: ColumnType) -> Description {
        let mut desc = Description::default();

        match column_type {
            ColumnType::Int64 => {
                let cd: &ColumnData<i64> = self.values().unwrap();
                describe_all!(rows, cd, desc, i64, DescriptionElement::Int);
            }
            ColumnType::Float64 => {
                let cd: &ColumnData<f64> = self.values().unwrap();
                describe_min_max!(rows, cd, desc, DescriptionElement::Float);
                let min = if let Some(DescriptionElement::Float(f)) = desc.get_min() {
                    f
                } else {
                    unreachable!() // by implementation
                };
                let max = if let Some(DescriptionElement::Float(f)) = desc.get_max() {
                    f
                } else {
                    unreachable!() // by implementation
                };
                let (rc, rt) = describe_top_n_f64(rows, cd, *min, *max);
                desc.count = rows.len();
                desc.unique_count = rc;
                desc.mode = Some(rt[0].0.clone());
                desc.top_n = Some(rt);

                describe_mean_deviation!(rows, cd, desc);
            }
            ColumnType::Enum => {
                let cd: &ColumnData<u32> = self.values().unwrap();
                describe_top_n!(rows, cd, desc, DescriptionElement::UInt);
            }
            ColumnType::Utf8 => {
                let cd: &ColumnData<String> = self.values().unwrap();
                describe_top_n!(rows, cd, desc, DescriptionElement::Text);
            }
            ColumnType::IpAddr => {
                let cd: &ColumnData<u32> = self.values().unwrap();
                let cd_ipaddr: ColumnData<IpAddr> = rows
                    .iter()
                    .map(|r| IpAddr::from(Ipv4Addr::from(cd[*r])))
                    .collect();
                let rows: Vec<usize> = rows.iter().enumerate().map(|(i, _r)| i).collect();
                describe_top_n!(&rows, &cd_ipaddr, desc, DescriptionElement::IpAddr);
            }
            ColumnType::DateTime => {
                let cd: &ColumnData<NaiveDateTime> = self.values().unwrap();
                let cd_only_date_hour: ColumnData<NaiveDateTime> = rows
                    .iter()
                    .map(|r| {
                        NaiveDateTime::new(
                            cd[*r].date(),
                            NaiveTime::from_hms(cd[*r].time().hour(), 0, 0),
                        )
                    })
                    .collect();
                let rows: Vec<usize> = rows.iter().enumerate().map(|(i, _r)| i).collect();
                describe_top_n!(
                    &rows,
                    &cd_only_date_hour,
                    desc,
                    DescriptionElement::DateTime
                );
            }
        }

        desc
    }
}

impl Clone for Column {
    fn clone(&self) -> Self {
        if self.inner.is::<ColumnData<i64>>() {
            let cd: &ColumnData<i64> = self.values().unwrap();
            Self {
                inner: Box::new(cd.clone()),
            }
        } else if self.inner.is::<ColumnData<f64>>() {
            let cd: &ColumnData<f64> = self.values().unwrap();
            Self {
                inner: Box::new(cd.clone()),
            }
        } else if self.inner.is::<ColumnData<u32>>() {
            let cd: &ColumnData<u32> = self.values().unwrap();
            Self {
                inner: Box::new(cd.clone()),
            }
        } else if self.inner.is::<ColumnData<String>>() {
            let cd: &ColumnData<String> = self.values().unwrap();
            Self {
                inner: Box::new(cd.clone()),
            }
        } else if self.inner.is::<ColumnData<NaiveDateTime>>() {
            let cd: &ColumnData<NaiveDateTime> = self.values().unwrap();
            Self {
                inner: Box::new(cd.clone()),
            }
        } else {
            panic!("invalid column type")
        }
    }
}

impl fmt::Debug for Column {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.inner.is::<ColumnData<i64>>() {
            let cd: &ColumnData<i64> = self.values().unwrap();
            write!(f, "{:?}", cd)
        } else if self.inner.is::<ColumnData<u32>>() {
            let cd: &ColumnData<u32> = self.values().unwrap();
            write!(f, "{:?}", cd)
        } else if self.inner.is::<ColumnData<f64>>() {
            let cd: &ColumnData<f64> = self.values().unwrap();
            write!(f, "{:?}", cd)
        } else if self.inner.is::<ColumnData<String>>() {
            let cd: &ColumnData<String> = self.values().unwrap();
            write!(f, "{:?}", cd)
        } else if self.inner.is::<ColumnData<NaiveDateTime>>() {
            let cd: &ColumnData<NaiveDateTime> = self.values().unwrap();
            write!(f, "{:?}", cd)
        } else {
            panic!("invalid column type")
        }
    }
}

impl PartialEq for Column {
    fn eq(&self, other: &Self) -> bool {
        if self.inner.is::<ColumnData<i64>>() {
            if !other.inner.is::<ColumnData<i64>>() {
                return false;
            }
            return self.inner.downcast_ref::<ColumnData<i64>>().unwrap()
                == other.inner.downcast_ref::<ColumnData<i64>>().unwrap();
        }
        if self.inner.is::<ColumnData<f64>>() {
            if !other.inner.is::<ColumnData<f64>>() {
                return false;
            }
            return self.inner.downcast_ref::<ColumnData<f64>>().unwrap()
                == other.inner.downcast_ref::<ColumnData<f64>>().unwrap();
        }
        if self.inner.is::<ColumnData<u32>>() {
            if !other.inner.is::<ColumnData<u32>>() {
                return false;
            }

            return self.inner.downcast_ref::<ColumnData<u32>>().unwrap()
                == other.inner.downcast_ref::<ColumnData<u32>>().unwrap();
        }
        if self.inner.is::<ColumnData<String>>() {
            if !other.inner.is::<ColumnData<String>>() {
                return false;
            }
            return self.inner.downcast_ref::<ColumnData<String>>().unwrap()
                == other.inner.downcast_ref::<ColumnData<String>>().unwrap();
        }
        if self.inner.is::<ColumnData<NaiveDateTime>>() {
            if !other.inner.is::<ColumnData<NaiveDateTime>>() {
                return false;
            }
            return self
                .inner
                .downcast_ref::<ColumnData<NaiveDateTime>>()
                .unwrap()
                == other
                    .inner
                    .downcast_ref::<ColumnData<NaiveDateTime>>()
                    .unwrap();
        }
        false
    }
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum DescriptionElement {
    Int(i64),
    UInt(u32),
    Enum(String), // describe() converts UInt -> Enum using enum maps. Without maps, by to_string().
    Float(f64),
    FloatRange(f64, f64),
    Text(String),
    IpAddr(IpAddr),
    DateTime(NaiveDateTime),
}

impl fmt::Display for DescriptionElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(x) => write!(f, "{}", x),
            Self::UInt(x) => write!(f, "{}", x),
            Self::Enum(x) | Self::Text(x) => write!(f, "{}", x),
            Self::Float(x) => write!(f, "{}", x),
            Self::FloatRange(x, y) => write!(f, "({} - {})", x, y),
            Self::IpAddr(x) => write!(f, "{}", x),
            Self::DateTime(x) => write!(f, "{}", x),
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Description {
    count: usize,
    unique_count: usize,
    mean: Option<f64>,
    s_deviation: Option<f64>,
    min: Option<DescriptionElement>,
    max: Option<DescriptionElement>,
    top_n: Option<Vec<(DescriptionElement, usize)>>,
    mode: Option<DescriptionElement>,
}

impl fmt::Display for Description {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Start of Description")?;
        writeln!(f, "   count: {}", self.count)?;
        writeln!(f, "   unique count: {}", self.unique_count)?;
        if self.mean.is_some() {
            writeln!(f, "   mean: {}", self.get_mean().unwrap())?;
        }
        if self.s_deviation.is_some() {
            writeln!(f, "   s-deviation: {}", self.get_s_deviation().unwrap())?;
        }
        if self.min.is_some() {
            writeln!(f, "   min: {}", self.get_min().unwrap())?;
        }
        if self.max.is_some() {
            writeln!(f, "   max: {}", self.get_max().unwrap())?;
        }
        writeln!(f, "   Top N")?;
        for (d, c) in self.get_top_n().unwrap() {
            writeln!(f, "      data: {}      count: {}", d, c)?;
        }
        if self.mode.is_some() {
            writeln!(f, "   mode: {}", self.get_mode().unwrap())?;
        }
        writeln!(f, "End of Description")
    }
}

impl fmt::Display for ColumnType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let display = match self {
            Self::Int64 => "Int64",
            Self::Float64 => "Float64",
            Self::Enum => "Enum",
            Self::Utf8 => "Utf8",
            Self::IpAddr => "IpAddr",
            Self::DateTime => "DateTime",
        };
        writeln!(f, "ColumnType::{}", display)
    }
}

impl Description {
    pub fn get_count(&self) -> usize {
        self.count
    }

    pub fn get_unique_count(&self) -> usize {
        self.unique_count
    }

    pub fn get_mean(&self) -> Option<f64> {
        self.mean
    }

    pub fn get_s_deviation(&self) -> Option<f64> {
        self.s_deviation
    }

    pub fn get_min(&self) -> Option<&DescriptionElement> {
        self.min.as_ref()
    }

    pub fn get_max(&self) -> Option<&DescriptionElement> {
        self.max.as_ref()
    }

    pub fn get_top_n(&self) -> Option<&Vec<(DescriptionElement, usize)>> {
        self.top_n.as_ref()
    }

    pub fn get_mode(&self) -> Option<&DescriptionElement> {
        self.mode.as_ref()
    }
}

#[allow(clippy::ptr_arg)]
fn count_sort<T: Clone + Eq + Hash>(rows: &[usize], cd: &ColumnData<T>) -> Vec<(T, usize)> {
    let mut count: HashMap<&T, usize> = HashMap::new();
    let mut top_n: Vec<(T, usize)> = Vec::new();
    for r in rows {
        let c = count.entry(&cd[*r]).or_insert(0);
        *c += 1;
    }
    for (k, v) in &count {
        top_n.push(((*k).clone(), *v));
    }
    top_n.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    top_n
}

#[allow(clippy::ptr_arg)]
fn describe_top_n_f64(
    rows: &[usize],
    cd: &ColumnData<f64>,
    min: f64,
    max: f64,
) -> (usize, Vec<(DescriptionElement, usize)>) {
    let interval: f64 = (max - min) / NUM_OF_FLOAT_INTERVALS.to_f64().expect("<= 100");
    let mut count: Vec<(usize, usize)> = vec![(0, 0); NUM_OF_FLOAT_INTERVALS];

    for (i, item) in count.iter_mut().enumerate().take(NUM_OF_FLOAT_INTERVALS) {
        item.0 = i;
    }

    for r in rows {
        let mut slot = ((cd[*r] - min) / interval)
            .floor()
            .to_usize()
            .expect("< 100");
        if slot == NUM_OF_FLOAT_INTERVALS {
            slot = NUM_OF_FLOAT_INTERVALS - 1;
        }
        count[slot].1 += 1;
    }

    count.retain(|&c| c.1 > 0);

    count.sort_unstable_by(|a, b| b.1.cmp(&a.1));

    let mut top_n: Vec<(DescriptionElement, usize)> = Vec::new();

    let num_top_n = if NUM_OF_TOP_N > count.len() {
        count.len()
    } else {
        NUM_OF_TOP_N
    };

    for item in count.iter().take(num_top_n) {
        if item.1 == 0 {
            break;
        }
        top_n.push((
            DescriptionElement::FloatRange(
                min + (item.0).to_f64().expect("< 30") * interval,
                min + (item.0 + 1).to_f64().expect("<= 30") * interval,
            ),
            item.1,
        ));
    }
    (count.len(), top_n)
}

#[allow(clippy::ptr_arg)]
fn find_min_max<T: PartialOrd + Clone>(rows: &[usize], cd: &ColumnData<T>) -> (T, T) {
    let mut min = cd[rows[0]].clone();
    let mut max = cd[rows[0]].clone();

    for r in rows {
        let data = &cd[*r];
        if min > *data {
            min = data.clone();
        }
        if max < *data {
            max = data.clone();
        }
    }
    (min, max)
}

macro_rules! column_from {
    ( $t:ty ) => {
        impl From<Vec<$t>> for Column {
            fn from(v: Vec<$t>) -> Self {
                Self { inner: Box::new(v) }
            }
        }
    };
}

column_from!(i64);
column_from!(f64);
column_from!(u32);
column_from!(String);
column_from!(NaiveDateTime);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Column;
    use chrono::{NaiveDate, NaiveDateTime};
    use std::convert::TryFrom;
    use std::net::Ipv4Addr;

    #[test]
    fn debug_column() {
        let col = Column::with_data(vec![64_i64]);
        assert_eq!(format!("{:?}", col), "[64]");

        let col = Column::with_data(vec![32_u32]);
        assert_eq!(format!("{:?}", col), "[32]");

        let col = Column::with_data(vec![64_f64]);
        assert_eq!(format!("{:?}", col), "[64.0]");

        let col = Column::with_data(vec!["string".to_string()]);
        assert_eq!(format!("{:?}", col), r#"["string"]"#);

        let col = Column::with_data(vec![NaiveDateTime::from_timestamp(999, 0)]);
        assert_eq!(format!("{:?}", col), "[1970-01-01T00:16:39]");
    }

    fn reverse_enum_maps(
        enum_maps: &HashMap<usize, HashMap<String, (u32, usize)>>,
    ) -> Arc<HashMap<usize, Arc<HashMap<u32, Vec<String>>>>> {
        Arc::new(
            enum_maps
                .iter()
                .filter_map(|(index, map)| {
                    if map.is_empty() {
                        None
                    } else {
                        let mut r_map_column = HashMap::<u32, Vec<String>>::new();
                        for (s, e) in map {
                            if let Some(v) = r_map_column.get_mut(&e.0) {
                                v.push(s.clone());
                            } else {
                                r_map_column.insert(e.0, vec![s.clone()]);
                            }
                        }
                        r_map_column.insert(0_u32, vec!["_Over One_".to_string()]); // unmapped ones.
                        r_map_column.insert(u32::max_value(), vec!["_Err_".to_string()]); // something wrong.
                        Some((*index, Arc::new(r_map_column)))
                    }
                })
                .collect(),
        )
    }

    #[test]
    fn description_test() {
        let c0_v: Vec<i64> = vec![1, 3, 3, 5, 2, 1, 3];
        let c1_v: Vec<String> = vec![
            "111a qwer".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
            "b".to_string(),
            "111a qwer".to_string(),
            "111a qwer".to_string(),
        ];
        let c2_v: Vec<u32> = vec![
            Ipv4Addr::new(127, 0, 0, 1).into(),
            Ipv4Addr::new(127, 0, 0, 2).into(),
            Ipv4Addr::new(127, 0, 0, 3).into(),
            Ipv4Addr::new(127, 0, 0, 4).into(),
            Ipv4Addr::new(127, 0, 0, 2).into(),
            Ipv4Addr::new(127, 0, 0, 2).into(),
            Ipv4Addr::new(127, 0, 0, 3).into(),
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

        let c0 = Column::from(c0_v);
        let c1 = Column::from(c1_v);
        let c2 = Column::from(c2_v);
        let c3 = Column::from(c3_v);
        let c4 = Column::from(c4_v);
        let c5 = Column::from(c5_v);
        let c_v: Vec<Column> = vec![c0, c1, c2, c3, c4, c5];
        let table = Table::try_from(c_v).expect("invalid columns");
        let column_types = Arc::new(vec![
            ColumnType::Int64,
            ColumnType::Utf8,
            ColumnType::IpAddr,
            ColumnType::Float64,
            ColumnType::DateTime,
            ColumnType::Enum,
        ]);
        let rows = vec![0_usize, 3, 1, 4, 2, 6, 5];
        let ds = table.describe(&rows, &column_types, &reverse_enum_maps(&HashMap::new()));

        assert_eq!(4, ds[0].unique_count);
        assert_eq!(
            DescriptionElement::Text("111a qwer".to_string()),
            *ds[1].get_mode().unwrap()
        );
        assert_eq!(
            DescriptionElement::IpAddr(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 3))),
            ds[2].get_top_n().unwrap()[1].0
        );
        assert_eq!(3, ds[3].unique_count);
        assert_eq!(
            DescriptionElement::DateTime(NaiveDate::from_ymd(2019, 9, 22).and_hms(6, 0, 0)),
            ds[4].get_top_n().unwrap()[0].0
        );
        assert_eq!(3, ds[5].unique_count);

        let mut c5_map: HashMap<u32, String> = HashMap::new();
        c5_map.insert(1, "t1".to_string());
        c5_map.insert(2, "t2".to_string());
        c5_map.insert(7, "t3".to_string());
        let mut labels = HashMap::new();
        labels.insert(5, c5_map.into_iter().map(|(k, v)| (v, (k, 0))).collect());
        let ds = table.describe(&rows, &column_types, &reverse_enum_maps(&labels));

        assert_eq!(4, ds[0].unique_count);
        assert_eq!(
            DescriptionElement::Text("111a qwer".to_string()),
            *ds[1].get_mode().unwrap()
        );
        assert_eq!(
            DescriptionElement::IpAddr(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 3))),
            ds[2].get_top_n().unwrap()[1].0
        );
        assert_eq!(3, ds[3].unique_count);
        assert_eq!(
            DescriptionElement::DateTime(NaiveDate::from_ymd(2019, 9, 22).and_hms(6, 0, 0)),
            ds[4].get_top_n().unwrap()[0].0
        );
        assert_eq!(
            DescriptionElement::Enum("t2".to_string()),
            *ds[5].get_mode().unwrap()
        );
    }
}
