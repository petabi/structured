use chrono::{NaiveDateTime, NaiveTime, Timelike};
use itertools::izip;
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use statistical::*;
use std::any::Any;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::fmt;
use std::hash::Hash;
use std::net::IpAddr;
use std::slice::Iter;

use crate::{DataType, Schema};

const NUM_OF_FLOAT_INTERVALS: usize = 100;
const NUM_OF_TOP_N: usize = 30;

type ColumnOfOneRow = DescriptionElement;

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
                DataType::Int => Column::new::<i64>(),
                DataType::Float => Column::new::<f64>(),
                DataType::Str => Column::new::<String>(),
                DataType::Enum => Column::new::<u32>(),
                DataType::IpAddr => Column::new::<IpAddr>(),
                DataType::DateTime => Column::new::<NaiveDateTime>(),
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
    ///
    /// # Panics
    ///
    /// Panics if there is no column.
    #[allow(dead_code)] // Used by tests only.
    pub fn num_rows(&self) -> usize {
        let col = &self.columns[0];
        col.len()
    }

    pub fn describe(&self, enum_maps: Option<&HashMap<usize, HashMap<String, u32>>>) -> Vec<Description> {
        self.columns.iter().enumerate().map(|(index, column)| {
            if column.inner.is::<ColumnData<u32>>() {
                let map = if let Some(enum_maps) = enum_maps {
                    if let Some(map) = enum_maps.get(&index) {
                        if !map.is_empty() {
                            Some(map)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };
                column.describe_enum(map)
            } else {
                column.describe()
            }
        }).collect()
    }

    pub fn get_index_of_event(&self, eventid: u64) -> Option<&usize> {
        self.event_ids.get(&eventid)
    }

    // TODO : add error handling
    pub fn push_one_row(
        &mut self,
        one_row: Vec<ColumnOfOneRow>,
        event_id: u64,
    ) -> Result<(), &'static str> {
        for (col, val) in izip!(self.columns.iter_mut(), one_row) {
            match val {
                ColumnOfOneRow::Int(v) => {
                    if let Some(c) = col.values_mut::<i64>() {
                        c.push(v)
                    };
                }
                ColumnOfOneRow::UInt(v) => {
                    if let Some(c) = col.values_mut::<u32>() {
                        c.push(v)
                    };
                }
                ColumnOfOneRow::Float(v) => {
                    if let Some(c) = col.values_mut::<f64>() {
                        c.push(v)
                    };
                }
                ColumnOfOneRow::Text(v) => {
                    if let Some(c) = col.values_mut::<String>() {
                        c.push(v)
                    };
                }
                ColumnOfOneRow::IpAddr(v) => {
                    if let Some(c) = col.values_mut::<IpAddr>() {
                        c.push(v)
                    };
                }
                ColumnOfOneRow::DateTime(v) => {
                    if let Some(c) = col.values_mut::<NaiveDateTime>() {
                        c.push(v)
                    };
                }
                _ => (),
            }
        }
        let len = self.event_ids.len();
        self.event_ids.entry(event_id).or_insert(len);
        Ok(())
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
    ( $cd:expr, $d:expr, $t1:ty, $t2:expr) => {
        describe_min_max!($cd, $d, $t2);
        describe_top_n!($cd, $d, $t2);
        describe_mean_deviation!($cd, $d);
    };
}

macro_rules! describe_min_max {
    ( $cd:expr, $d:expr, $t2:expr ) => {{
        let (min, max) = find_min_max($cd);
        $d.min = Some($t2(min));
        $d.max = Some($t2(max));
    }};
}

macro_rules! describe_mean_deviation {
    ( $cd:expr, $d:expr ) => {
        let vf: Vec<f64> = $cd.iter().map(|x| *x as f64).collect();
        let m = mean(&vf);
        $d.mean = Some(m);
        $d.s_deviation = Some(population_standard_deviation(&vf, Some(m)));
    };
}

macro_rules! describe_top_n {
    ( $cd:expr, $d:expr, $t2:expr ) => {
        let top_n_native = count_sort($cd);
        $d.count = $cd.len();
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

#[derive(Debug)]
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
        } else if self.inner.is::<ColumnData<IpAddr>>() {
            column_len!(self, IpAddr);
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
        } else if self.inner.is::<ColumnData<IpAddr>>() {
            column_append!(self.inner, other.inner, IpAddr);
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

    pub fn describe_enum(&self, enum_map: Option<&HashMap<String, u32>>) -> Description {
        let desc = self.describe();
        let is_map = if let Some(enum_map) = enum_map {
            if !enum_map.is_empty() {
                true
            } else {
                false
            }
        } else {
            false
        };

        let (top_n, mode) = {
            if is_map {
                let mut reverted_map = HashMap::new();
                for (k, v) in enum_map.expect("safe").iter() {
                    reverted_map.insert(*v, k);
                }
                (match desc.get_top_n() {
                    Some(top_n) => Some(top_n.iter().map(|(v, c)| {
                        if let DescriptionElement::UInt(value) = v {
                            (DescriptionElement::Enum(reverted_map.get(value).unwrap().to_string()), *c)
                        } else {
                            (DescriptionElement::Enum("_N/A_".to_string()), *c)
                        }
                    }).collect()),
                    None => None,
                }, 
                match desc.get_mode() {
                    Some(mode) => {
                        if let DescriptionElement::UInt(value) = mode {
                            Some(DescriptionElement::Enum(reverted_map.get(&value).unwrap().to_string()))
                        } else { None }
                    }
                    None => None,
                })
            } else {
                (match desc.get_top_n() {
                    Some(top_n) => Some(top_n.iter().map(|(v, c)| {
                        if let DescriptionElement::UInt(value) = v {
                            (DescriptionElement::Enum(value.to_string()), *c)
                        } else {
                            (DescriptionElement::Enum("_N/A_".to_string()), *c)
                        }
                    }).collect()),
                    None => None,
                },
                match desc.get_mode() {
                    Some(mode) => {
                        if let DescriptionElement::UInt(value) = mode {
                            Some(DescriptionElement::Enum(value.to_string()))
                        } else { None }
                    }
                    None => None,
                })
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

    pub fn describe(&self) -> Description {
        let mut desc = Description::default();

        if self.inner.is::<ColumnData<i64>>() {
            let cd: &ColumnData<i64> = self.values().unwrap();
            describe_all!(cd, desc, i64, DescriptionElement::Int);
        } else if self.inner.is::<ColumnData<f64>>() {
            let cd: &ColumnData<f64> = self.values().unwrap();
            describe_min_max!(cd, desc, DescriptionElement::Float);

            let min = match desc.get_min().unwrap() {
                DescriptionElement::Float(f) => f,
                _ => panic!(), // TODO: add error handling
            };

            let max = match desc.get_max().unwrap() {
                DescriptionElement::Float(f) => f,
                _ => panic!(), // TODO: add error handling
            };
            let (rc, rt) = describe_top_n_f64(cd, *min, *max);
            desc.count = cd.len();
            desc.unique_count = rc;
            desc.mode = Some(rt[0].0.clone());
            desc.top_n = Some(rt);

            describe_mean_deviation!(cd, desc);
        } else if self.inner.is::<ColumnData<u32>>() {
            let cd: &ColumnData<u32> = self.values().unwrap();
            describe_top_n!(cd, desc, DescriptionElement::UInt);
        } else if self.inner.is::<ColumnData<String>>() {
            let cd: &ColumnData<String> = self.values().unwrap();
            describe_top_n!(cd, desc, DescriptionElement::Text);
        } else if self.inner.is::<ColumnData<IpAddr>>() {
            let cd: &ColumnData<IpAddr> = self.values().unwrap();
            describe_top_n!(cd, desc, DescriptionElement::IpAddr);
        } else if self.inner.is::<ColumnData<NaiveDateTime>>() {
            let cd: &ColumnData<NaiveDateTime> = self.values().unwrap();

            let cd_onlytime: ColumnData<NaiveDateTime> = cd
                .iter()
                .map(|dt| {
                    NaiveDateTime::new(dt.date(), NaiveTime::from_hms(dt.time().hour(), 0, 0))
                })
                .collect();
            describe_top_n!(&cd_onlytime, desc, DescriptionElement::DateTime);
        } else {
            panic!("invalid column type");
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
        } else if self.inner.is::<ColumnData<IpAddr>>() {
            let cd: &ColumnData<IpAddr> = self.values().unwrap();
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
        if self.inner.is::<ColumnData<IpAddr>>() {
            if !other.inner.is::<ColumnData<IpAddr>>() {
                return false;
            }
            return self.inner.downcast_ref::<ColumnData<IpAddr>>().unwrap()
                == other.inner.downcast_ref::<ColumnData<IpAddr>>().unwrap();
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
    UInt(u32), // only internal use because raw data stored as u32
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
            Self::Enum(x) => write!(f, "{}", x),
            Self::Float(x) => write!(f, "{}", x),
            Self::FloatRange(x, y) => write!(f, "({} - {})", x, y),
            Self::Text(x) => write!(f, "{}", x),
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
fn count_sort<T: Clone + Eq + Hash>(cd: &ColumnData<T>) -> Vec<(T, usize)> {
    let mut count: HashMap<&T, usize> = HashMap::new();
    let mut top_n: Vec<(T, usize)> = Vec::new();
    for i in cd {
        let c = count.entry(i).or_insert(0);
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
    cd: &ColumnData<f64>,
    min: f64,
    max: f64,
) -> (usize, Vec<(DescriptionElement, usize)>) {
    let interval: f64 = (max - min) / NUM_OF_FLOAT_INTERVALS.to_f64().expect("<= 100");
    let mut count: Vec<(usize, usize)> = vec![(0, 0); NUM_OF_FLOAT_INTERVALS];

    for (i, item) in count.iter_mut().enumerate().take(NUM_OF_FLOAT_INTERVALS) {
        item.0 = i;
    }

    for d in cd.iter() {
        let mut slot = ((d - min) / interval).floor().to_usize().expect("< 100");
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
fn find_min_max<T: PartialOrd + Clone>(cd: &ColumnData<T>) -> (T, T) {
    let mut min = cd.first().unwrap();
    let mut max = cd.first().unwrap();

    for i in cd.iter() {
        if min > i {
            min = i;
        }
        if max < i {
            max = i;
        }
    }
    ((*min).clone(), (*max).clone())
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
column_from!(IpAddr);
column_from!(NaiveDateTime);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Column;
    use chrono::{NaiveDate, NaiveDateTime};
    use std::convert::TryFrom;
    use std::net::Ipv4Addr;

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
        let c2_v: Vec<IpAddr> = vec![
            IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
            IpAddr::V4(Ipv4Addr::new(127, 0, 0, 2)),
            IpAddr::V4(Ipv4Addr::new(127, 0, 0, 3)),
            IpAddr::V4(Ipv4Addr::new(127, 0, 0, 4)),
            IpAddr::V4(Ipv4Addr::new(127, 0, 0, 2)),
            IpAddr::V4(Ipv4Addr::new(127, 0, 0, 2)),
            IpAddr::V4(Ipv4Addr::new(127, 0, 0, 3)),
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
        let table_org = Table::try_from(c_v).expect("invalid columns");
        let table = table_org.clone();
        move_table_add_row(table_org);
        let ds = table.describe(None);

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
    }

    pub fn move_table_add_row(mut table: Table) {
        let one_row: Vec<ColumnOfOneRow> = vec![
            ColumnOfOneRow::Int(3),
            ColumnOfOneRow::Text("Hundred".to_string()),
            ColumnOfOneRow::IpAddr(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 100))),
            ColumnOfOneRow::Float(100.100),
            ColumnOfOneRow::DateTime(NaiveDate::from_ymd(2019, 10, 10).and_hms(10, 10, 10)),
        ];
        table
            .push_one_row(one_row, 1)
            .expect("Failure in adding a row");
        let ds = table.describe(None);
        assert_eq!(DescriptionElement::Int(3), ds[0].get_top_n().unwrap()[0].0);
        assert_eq!(4, ds[0].get_top_n().unwrap()[0].1);
    }
}
