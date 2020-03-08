use crate::array::*;
use crate::datatypes::*;
use crate::memory::AllocationError;
use crate::{DataType, Schema};
use chrono::NaiveDateTime;
use dashmap::DashMap;
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use statistical::*;
use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::fmt;
use std::hash::Hash;
use std::iter::{Flatten, Iterator};
use std::marker::PhantomData;
use std::net::{IpAddr, Ipv4Addr};
use std::ops::{Deref, Index};
use std::slice;
use std::sync::Arc;
use std::vec;
use strum_macros::EnumString;

pub const DEFAULT_NUM_OF_TOP_N: u32 = 30;
const DEFAULT_NUM_OF_TOP_N_OF_DATETIME: u32 = 336; // 24 hours x 14 days
const NUM_OF_FLOAT_INTERVALS: usize = 100;
const DEFAULT_TIME_INTERVAL: u32 = 3600; // seconds
const MAX_TIME_INTERVAL: u32 = 86_400; // one day
const MIN_TIME_INTERVAL: u32 = 30;

type ConcurrentEnumMaps = Arc<DashMap<usize, Arc<DashMap<String, (u32, usize)>>>>;
type ReverseEnumMaps = Arc<HashMap<usize, Arc<HashMap<u32, Vec<String>>>>>;

/// The data type of a table column.
#[derive(Clone, Copy, Debug, Deserialize, EnumString, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "snake_case")]
pub enum ColumnType {
    Int64,
    Float64,
    DateTime,
    IpAddr,
    Enum,
    Utf8,
    Binary,
}

impl Into<DataType> for ColumnType {
    #[must_use]
    fn into(self) -> DataType {
        match self {
            Self::Int64 => DataType::Int64,
            Self::Float64 => DataType::Float64,
            Self::DateTime => DataType::Timestamp(TimeUnit::Second),
            Self::Enum | Self::IpAddr => DataType::UInt32,
            Self::Utf8 => DataType::Utf8,
            Self::Binary => DataType::Binary,
        }
    }
}

/// Structured data represented in a column-oriented form.
#[derive(Debug, Default, Clone)]
pub struct Table {
    schema: Arc<Schema>,
    columns: Vec<Column>,
    event_ids: HashMap<u64, usize>,
}

impl Table {
    /// Creates a new `Table` with the given `schema` and `columns`.
    ///
    /// # Errors
    ///
    /// Returns an error if `columns` have different lengths.
    pub fn new(
        schema: Arc<Schema>,
        columns: Vec<Column>,
        event_ids: HashMap<u64, usize>,
    ) -> Result<Self, &'static str> {
        let len = if let Some(col) = columns.first() {
            col.len()
        } else {
            return Ok(Self {
                schema,
                columns,
                event_ids: HashMap::new(),
            });
        };
        if columns.iter().skip(1).all(|c| c.len() == len) {
            Ok(Self {
                schema,
                columns,
                event_ids,
            })
        } else {
            Err("columns must have the same length")
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
    #[must_use]
    pub fn columns(&self) -> slice::Iter<Column> {
        self.columns.iter()
    }

    /// Returns the number of columns in the table.
    #[must_use]
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Returns the number of rows in the table.
    #[must_use]
    pub fn num_rows(&self) -> usize {
        if self.columns.is_empty() {
            0_usize
        } else {
            let col = &self.columns[0];
            col.len()
        }
    }

    #[must_use]
    pub fn get_statistics(
        &self,
        rows: &[usize],
        column_types: &Arc<Vec<ColumnType>>,
        r_enum_maps: &ReverseEnumMaps,
        time_intervals: &Arc<Vec<u32>>,
        numbers_of_top_n: &Arc<Vec<u32>>,
    ) -> Vec<ColumnStatistics> {
        self.columns
            .iter()
            .enumerate()
            .map(|(index, column)| {
                let description = column.describe(rows, column_types[index]);
                let n_largest_count = if let ColumnType::Enum = column_types[index] {
                    column.get_n_largest_count_enum(
                        rows,
                        r_enum_maps.get(&index).unwrap_or(&Arc::new(HashMap::new())),
                        *numbers_of_top_n.get(index).unwrap_or(&DEFAULT_NUM_OF_TOP_N),
                    )
                } else if let ColumnType::DateTime = column_types[index] {
                    let mut cn = 0_usize;
                    for i in 0..index {
                        if let ColumnType::DateTime = column_types[i] {
                            cn += 1;
                        }
                    }
                    column.get_n_larget_count_datetime(
                        rows,
                        *time_intervals.get(cn).unwrap_or(&DEFAULT_TIME_INTERVAL),
                        *numbers_of_top_n
                            .get(index)
                            .unwrap_or(&DEFAULT_NUM_OF_TOP_N_OF_DATETIME),
                    )
                } else if let ColumnType::Float64 = column_types[index] {
                    if let (Some(Element::Float(min)), Some(Element::Float(max))) =
                        (description.get_min(), description.get_max())
                    {
                        column.get_n_largest_count_float64(
                            rows,
                            *numbers_of_top_n.get(index).unwrap_or(&DEFAULT_NUM_OF_TOP_N),
                            *min,
                            *max,
                        )
                    } else {
                        NLargestCount::default()
                    }
                } else {
                    column.get_n_largest_count(
                        rows,
                        column_types[index],
                        *numbers_of_top_n.get(index).unwrap_or(&DEFAULT_NUM_OF_TOP_N),
                    )
                };

                ColumnStatistics {
                    description,
                    n_largest_count,
                }
            })
            .collect()
    }

    #[must_use]
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
            self.limit_enum_values(*column_index, &mapped_enums)
                .unwrap();
        }
        (enum_portion_dimensions, enum_set_dimensions)
    }

    fn limit_enum_values(
        &mut self,
        column_index: usize,
        mapped_enums: &HashSet<u32>,
    ) -> Result<(), AllocationError> {
        let col = &mut self.columns[column_index];
        let mut builder = primitive::Builder::<UInt32Type>::with_capacity(col.len())?;
        for val in col.iter::<UInt32ArrayType>().unwrap() {
            let new_val = if mapped_enums.contains(val) {
                *val
            } else {
                0_u32 // if unmapped out of the predefined rate, enum value set to 0_u32.
            };
            builder.try_push(new_val)?;
        }
        col.arrays.clear();
        col.arrays.push(builder.build());
        Ok(())
    }
}

macro_rules! describe_min_max {
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

macro_rules! describe_mean_deviation {
    ( $vf:expr, $t1:ty, $d:expr ) => {
        let m = mean(&$vf);
        $d.mean = Some(m);
        $d.s_deviation = Some(population_standard_deviation(&$vf, Some(m)));
    };
}

macro_rules! describe_top_n {
    ( $iter:expr, $len:expr, $d:expr, $t1:ty, $t2:expr, $num_of_top_n:expr ) => {
        let top_n_native: Vec<(&$t1, usize)> = count_sort($iter);
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
        $d.mode = Some(top_n[0].value.clone());
        $d.top_n = Some(top_n);
    };
}

/// A single column in a table.
#[derive(Clone, Debug, Default)]
pub struct Column {
    arrays: Vec<Arc<dyn Array>>,
    cumlen: Vec<usize>,
    len: usize,
}

impl Column {
    /// Converts a slice into a `Column`.
    ///
    /// # Errors
    ///
    /// Returns an error if memory allocation failed.
    pub fn try_from_slice<T>(slice: &[T::Native]) -> Result<Self, AllocationError>
    where
        T: PrimitiveType,
    {
        let array: Arc<dyn Array> = Arc::new(TryInto::<primitive::Array<T>>::try_into(slice)?);
        Ok(array.into())
    }

    fn len(&self) -> usize {
        self.len
    }

    fn try_get<'a, A, T>(&self, index: usize) -> Result<Option<&T>, TypeError>
    where
        A: ArrayType<Elem = &'a T>,
        A::Array: Index<usize, Output = T> + 'static,
        T: ?Sized + 'static,
    {
        if index >= self.len() {
            return Ok(None);
        }
        let (array_index, inner_index) = match self.cumlen.binary_search(&index) {
            Ok(i) => (i, 0),
            Err(i) => (i - 1, index - self.cumlen[i - 1]),
        };
        let typed_arr =
            if let Some(arr) = self.arrays[array_index].as_any().downcast_ref::<A::Array>() {
                arr
            } else {
                return Err(TypeError());
            };
        Ok(Some(typed_arr.index(inner_index)))
    }

    fn append(&mut self, other: &mut Self) {
        // TODO: make sure the types match
        self.arrays.append(&mut other.arrays);
        let len = self.len;
        self.cumlen
            .extend(other.cumlen.iter().skip(1).map(|v| v + len));
        self.len += other.len;
        other.len = 0;
    }

    /// Creates an iterator iterating over all the cells in this `Column`.
    ///
    /// # Errors
    ///
    /// Returns an error if the type parameter does not match with the type of
    /// this `Column`.
    pub fn iter<'a, T>(&'a self) -> Result<Flatten<vec::IntoIter<&'a T::Array>>, TypeError>
    where
        T: ArrayType,
        T::Array: 'static,
        &'a T::Array: IntoIterator,
    {
        let mut arrays: Vec<&T::Array> = Vec::with_capacity(self.arrays.len());
        for arr in &self.arrays {
            let typed_arr = if let Some(arr) = arr.as_any().downcast_ref::<T::Array>() {
                arr
            } else {
                return Err(TypeError());
            };
            arrays.push(typed_arr);
        }
        Ok(arrays.into_iter().flatten())
    }

    /// Creates an iterator iterating over a subset of the cells in this
    /// `Column`, designated by `selected`.
    ///
    /// # Errors
    ///
    /// Returns an error if the type parameter does not match with the type of
    /// this `Column`.
    pub fn view_iter<'a, 'b, A, T>(
        &'a self,
        selected: &'b [usize],
    ) -> Result<ViewIter<'a, 'b, A, T>, TypeError>
    where
        A: ArrayType<Elem = &'a T>,
        A::Array: Index<usize, Output = T> + 'static,
        T: ?Sized + 'static,
    {
        let mut arrays: Vec<&A::Array> = Vec::with_capacity(self.arrays.len());
        for arr in &self.arrays {
            let typed_arr = if let Some(arr) = arr.as_any().downcast_ref::<A::Array>() {
                arr
            } else {
                return Err(TypeError());
            };
            arrays.push(typed_arr);
        }
        Ok(ViewIter::new(self, selected.iter()))
    }

    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn get_n_largest_count_enum(
        &self,
        rows: &[usize],
        reverse_map: &Arc<HashMap<u32, Vec<String>>>,
        number_of_top_n: u32,
    ) -> NLargestCount {
        let desc = self.get_n_largest_count(rows, ColumnType::Enum, number_of_top_n);

        let (top_n, mode) = {
            if reverse_map.is_empty() {
                (
                    match desc.get_top_n() {
                        Some(top_n) => Some(
                            top_n
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
                        ),
                        None => None,
                    },
                    match desc.get_mode() {
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
                    match desc.get_top_n() {
                        Some(top_n) => Some(
                            top_n
                                .iter()
                                .map(|elem| {
                                    if let Element::UInt(value) = elem.value {
                                        (ElementCount {
                                            value: Element::Enum(reverse_map.get(&value).map_or(
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
                                            )),
                                            count: elem.count,
                                        })
                                    } else {
                                        ElementCount {
                                            value: Element::Enum("_N/A_".to_string()),
                                            count: elem.count,
                                        }
                                    }
                                })
                                .collect(),
                        ),
                        None => None,
                    },
                    match desc.get_mode() {
                        Some(mode) => {
                            if let Element::UInt(value) = mode {
                                Some(Element::Enum(reverse_map.get(value).map_or(
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

        NLargestCount {
            number_of_elements: desc.get_number_of_elements(),
            top_n,
            mode,
        }
    }

    #[must_use]
    pub fn describe(&self, rows: &[usize], column_type: ColumnType) -> Description {
        let mut desc = Description::default();

        desc.count = rows.len();
        match column_type {
            ColumnType::Int64 => {
                let iter = self.view_iter::<Int64ArrayType, i64>(rows).unwrap();
                describe_min_max!(iter, desc, Element::Int);
                let iter = self.view_iter::<Int64ArrayType, i64>(rows).unwrap();
                #[allow(clippy::cast_precision_loss)] // 52-bit precision is good enough
                let f_values: Vec<f64> = iter.map(|v: &i64| *v as f64).collect();
                describe_mean_deviation!(f_values, i64, desc);
            }
            ColumnType::Float64 => {
                let iter = self.view_iter::<Float64ArrayType, f64>(rows).unwrap();
                describe_min_max!(iter, desc, Element::Float);
                let iter = self.view_iter::<Float64ArrayType, f64>(rows).unwrap();
                let values = iter.cloned().collect::<Vec<_>>();
                describe_mean_deviation!(values, f64, desc);
            }
            _ => (),
        }

        desc
    }

    #[must_use]
    pub fn get_n_largest_count(
        &self,
        rows: &[usize],
        column_type: ColumnType,
        number_of_top_n: u32,
    ) -> NLargestCount {
        let mut desc = NLargestCount::default();

        match column_type {
            ColumnType::Int64 => {
                let iter = self.view_iter::<Int64ArrayType, i64>(rows).unwrap();
                describe_top_n!(iter, rows.len(), desc, i64, Element::Int, number_of_top_n);
            }
            ColumnType::Enum => {
                let iter = self.view_iter::<UInt32ArrayType, u32>(rows).unwrap();
                describe_top_n!(iter, rows.len(), desc, u32, Element::UInt, number_of_top_n);
            }
            ColumnType::Utf8 => {
                let iter = self.view_iter::<Utf8ArrayType, str>(rows).unwrap();
                describe_top_n!(iter, rows.len(), desc, str, Element::Text, number_of_top_n);
            }
            ColumnType::Binary => {
                let iter = self.view_iter::<BinaryArrayType, [u8]>(rows).unwrap();
                describe_top_n!(
                    iter,
                    rows.len(),
                    desc,
                    [u8],
                    Element::Binary,
                    number_of_top_n
                );
            }
            ColumnType::IpAddr => {
                let values = self
                    .view_iter::<UInt32ArrayType, u32>(rows)
                    .unwrap()
                    .map(|v: &u32| IpAddr::from(Ipv4Addr::from(*v)))
                    .collect::<Vec<_>>();
                describe_top_n!(
                    values.iter(),
                    rows.len(),
                    desc,
                    IpAddr,
                    Element::IpAddr,
                    number_of_top_n
                );
            }
            ColumnType::DateTime | ColumnType::Float64 => unreachable!(), // by implementation
        }

        desc
    }

    #[must_use]
    fn get_n_largest_count_float64(
        &self,
        rows: &[usize],
        number_of_top_n: u32,
        min: f64,
        max: f64,
    ) -> NLargestCount {
        let mut desc = NLargestCount::default();

        let iter = self.view_iter::<Float64ArrayType, f64>(rows).unwrap();
        let (rc, rt) = describe_top_n_f64(iter, min, max, number_of_top_n);
        desc.number_of_elements = rc;
        desc.mode = Some(rt[0].value.clone());
        desc.top_n = Some(rt);

        desc
    }

    #[must_use]
    fn get_n_larget_count_datetime(
        &self,
        rows: &[usize],
        time_interval: u32,
        number_of_top_n: u32,
    ) -> NLargestCount {
        let mut desc = NLargestCount::default();

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

        let values = self
            .view_iter::<Int64ArrayType, i64>(rows)
            .unwrap()
            .map(|v: &i64| {
                // The first interval of each day should start with 00:00:00.
                let mut ts = *v / (24 * 60 * 60) * (24 * 60 * 60);
                ts += (*v - ts) / time_interval * time_interval;
                NaiveDateTime::from_timestamp(ts, 0)
            })
            .collect::<Vec<_>>();

        describe_top_n!(
            values.iter(),
            rows.len(),
            desc,
            NaiveDateTime,
            Element::DateTime,
            number_of_top_n
        );
        // TODO: rename desc
        desc
    }
}

impl PartialEq for Column {
    #[must_use]
    fn eq(&self, other: &Self) -> bool {
        let data_type = match (self.arrays.first(), other.arrays.first()) {
            (Some(x_arr), Some(y_arr)) => {
                if x_arr.data().data_type() == y_arr.data().data_type() {
                    x_arr.data().data_type()
                } else {
                    return false;
                }
            }
            (Some(_), None) | (None, Some(_)) => return false,
            (None, None) => return true,
        };
        if self.len() != other.len() {
            return false;
        }

        match data_type {
            DataType::Int32 => self
                .iter::<Int32ArrayType>()
                .expect("invalid array")
                .zip(other.iter::<Int32ArrayType>().expect("invalid array"))
                .all(|(x, y)| x == y),
            DataType::Int64 | DataType::Timestamp(TimeUnit::Second) => self
                .iter::<Int64ArrayType>()
                .expect("invalid array")
                .zip(other.iter::<Int64ArrayType>().expect("invalid array"))
                .all(|(x, y)| x == y),
            DataType::UInt8 => self
                .iter::<UInt8ArrayType>()
                .expect("invalid array")
                .zip(other.iter::<UInt8ArrayType>().expect("invalid array"))
                .all(|(x, y)| x == y),
            DataType::UInt32 => self
                .iter::<UInt32ArrayType>()
                .expect("invalid array")
                .zip(other.iter::<UInt32ArrayType>().expect("invalid array"))
                .all(|(x, y)| x == y),
            DataType::Float64 => self
                .iter::<Float64ArrayType>()
                .expect("invalid array")
                .zip(other.iter::<Float64ArrayType>().expect("invalid array"))
                .all(|(x, y)| x == y),
            DataType::Utf8 => self
                .iter::<Utf8ArrayType>()
                .expect("invalid array")
                .zip(other.iter::<Utf8ArrayType>().expect("invalid array"))
                .all(|(x, y)| x == y),
            DataType::Binary => self
                .iter::<BinaryArrayType>()
                .expect("invalid array")
                .zip(other.iter::<BinaryArrayType>().expect("invalid array"))
                .all(|(x, y)| x == y),
        }
    }
}

impl From<Arc<dyn Array>> for Column {
    #[must_use]
    fn from(array: Arc<dyn Array>) -> Self {
        let len = array.len();
        Self {
            arrays: vec![array],
            cumlen: vec![0, len],
            len,
        }
    }
}

pub trait ArrayType {
    type Array: Array;
    type Elem;
}

macro_rules! make_array_type {
    ($(#[$outer:meta])*
    $name:ident, $array_ty:ty, $elem_ty:ty) => {
        $(#[$outer])*
        pub struct $name<'a> {
            _marker: PhantomData<&'a u8>,
        }

        impl<'a> ArrayType for $name<'a> {
            type Array = $array_ty;
            type Elem = &'a $elem_ty;
        }
    };
}

make_array_type!(
    /// Data type of a dynamic array whose elements are `i32`s.
    Int32ArrayType,
    primitive::Array<Int32Type>,
    i32
);
make_array_type!(
    /// Data type of a dynamic array whose elements are `i64`s.
    Int64ArrayType,
    primitive::Array<Int64Type>,
    i64
);
make_array_type!(
    /// Data type of a dynamic array whose elements are `u8`s.
    UInt8ArrayType,
    primitive::Array<UInt8Type>,
    u8
);
make_array_type!(
    /// Data type of a dynamic array whose elements are `u32`s.
    UInt32ArrayType,
    primitive::Array<UInt32Type>,
    u32
);
make_array_type!(
    /// Data type of a dynamic array whose elements are `f64`s.
    Float64ArrayType,
    primitive::Array<Float64Type>,
    f64
);
make_array_type!(
    /// Data type of a dynamic array whose elements are UTF-8 strings.
    Utf8ArrayType,
    StringArray,
    str
);
make_array_type!(
    /// Data type of a dynamic array whose elements are byte sequences.
    BinaryArrayType,
    BinaryArray,
    [u8]
);

#[derive(Debug, PartialEq)]
pub struct TypeError();

pub struct ViewIter<'a, 'b, A, T: ?Sized> {
    column: &'a Column,
    selected: slice::Iter<'b, usize>,
    _a_marker: PhantomData<A>,
    _t_marker: PhantomData<T>,
}

impl<'a, 'b, A, T> ViewIter<'a, 'b, A, T>
where
    A: ArrayType<Elem = &'a T>,
    A::Array: Index<usize, Output = T> + 'static,
    T: ?Sized + 'static,
{
    fn new(column: &'a Column, selected: slice::Iter<'b, usize>) -> Self {
        Self {
            column,
            selected,
            _a_marker: PhantomData,
            _t_marker: PhantomData,
        }
    }
}

impl<'a, 'b, A, T> Iterator for ViewIter<'a, 'b, A, T>
where
    A: ArrayType<Elem = &'a T>,
    A::Array: Index<usize, Output = T> + 'static,
    T: ?Sized + 'static,
{
    type Item = A::Elem;

    fn next(&mut self) -> Option<Self::Item> {
        let selected = if let Some(selected) = self.selected.next() {
            selected
        } else {
            return None;
        };
        if let Ok(elem) = self.column.try_get::<A, T>(*selected) {
            elem
        } else {
            None
        }
    }
}

/// The underlying data type of a column description.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum Element {
    Int(i64),
    UInt(u32),
    Enum(String), // describe() converts UInt -> Enum using enum maps. Without maps, by to_string().
    Float(f64),
    FloatRange(FloatRange),
    Text(String),
    Binary(Vec<u8>),
    IpAddr(IpAddr),
    DateTime(NaiveDateTime),
}

#[derive(Debug, Default, PartialEq, Clone, Serialize, Deserialize)]
pub struct FloatRange {
    smallest: f64,
    largest: f64,
}

impl fmt::Display for Element {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(x) => write!(f, "{}", x),
            Self::UInt(x) => write!(f, "{}", x),
            Self::Enum(x) | Self::Text(x) => write!(f, "{}", x),
            Self::Binary(x) => write!(f, "{:#?}", x),
            Self::Float(x) => write!(f, "{}", x),
            Self::FloatRange(x) => write!(f, "({} - {})", x.smallest, x.largest),
            Self::IpAddr(x) => write!(f, "{}", x),
            Self::DateTime(x) => write!(f, "{}", x),
        }
    }
}

/// Statistical summary of data of the same type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnStatistics {
    description: Description,
    n_largest_count: NLargestCount,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Description {
    count: usize,
    mean: Option<f64>,
    s_deviation: Option<f64>,
    min: Option<Element>,
    max: Option<Element>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElementCount {
    value: Element,
    count: usize,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct NLargestCount {
    number_of_elements: usize,
    top_n: Option<Vec<ElementCount>>,
    mode: Option<Element>,
}

impl fmt::Display for Description {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Start of Description")?;
        writeln!(f, "   count: {}", self.count)?;
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
        writeln!(f, "End of Description")
    }
}

impl fmt::Display for NLargestCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Start of NLargestCount")?;
        writeln!(
            f,
            "   number of elements: {}",
            self.get_number_of_elements()
        )?;
        writeln!(f, "   Top N")?;
        for elem in self.get_top_n().unwrap() {
            writeln!(f, "      data: {}      count: {}", elem.value, elem.count)?;
        }
        if self.mode.is_some() {
            writeln!(f, "   mode: {}", self.get_mode().unwrap())?;
        }
        writeln!(f, "End of NLargestCount")
    }
}

impl fmt::Display for ColumnType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let display = match self {
            Self::Int64 => "Int64",
            Self::Float64 => "Float64",
            Self::Enum => "Enum",
            Self::Utf8 => "Utf8",
            Self::Binary => "Binary",
            Self::IpAddr => "IpAddr",
            Self::DateTime => "DateTime",
        };
        writeln!(f, "ColumnType::{}", display)
    }
}

impl Description {
    #[must_use]
    pub fn get_count(&self) -> usize {
        self.count
    }

    #[must_use]
    pub fn get_mean(&self) -> Option<f64> {
        self.mean
    }

    #[must_use]
    pub fn get_s_deviation(&self) -> Option<f64> {
        self.s_deviation
    }

    #[must_use]
    pub fn get_min(&self) -> Option<&Element> {
        self.min.as_ref()
    }

    #[must_use]
    pub fn get_max(&self) -> Option<&Element> {
        self.max.as_ref()
    }
}

impl NLargestCount {
    #[must_use]
    pub fn get_number_of_elements(&self) -> usize {
        self.number_of_elements
    }

    #[must_use]
    pub fn get_top_n(&self) -> Option<&Vec<ElementCount>> {
        self.top_n.as_ref()
    }

    #[must_use]
    pub fn get_mode(&self) -> Option<&Element> {
        self.mode.as_ref()
    }
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

fn describe_top_n_f64<I>(
    iter: I,
    min: f64,
    max: f64,
    number_of_top_n: u32,
) -> (usize, Vec<ElementCount>)
where
    I: Iterator,
    I::Item: Deref<Target = f64>,
{
    let interval: f64 = (max - min) / NUM_OF_FLOAT_INTERVALS.to_f64().expect("<= 100");
    let mut count: Vec<(usize, usize)> = vec![(0, 0); NUM_OF_FLOAT_INTERVALS];

    for (i, item) in count.iter_mut().enumerate().take(NUM_OF_FLOAT_INTERVALS) {
        item.0 = i;
    }

    for v in iter {
        let mut slot = ((*v - min) / interval).floor().to_usize().expect("< 100");
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
fn find_min_max<I, T>(mut iter: I) -> Option<MinMax<<I::Item as Deref>::Target>>
where
    I: Iterator,
    I::Item: Deref<Target = T>,
    T: PartialOrd + Clone,
{
    let mut min = if let Some(first) = iter.next() {
        (*first).clone()
    } else {
        return None;
    };
    let mut max = min.clone();

    for v in iter {
        if min > *v {
            min = (*v).clone();
        } else if max < *v {
            max = (*v).clone();
        }
    }
    Some(MinMax { min, max })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Column;
    use chrono::NaiveDate;
    use std::convert::TryFrom;
    use std::net::Ipv4Addr;

    #[test]
    fn table_try_default() {
        let table = Table::default();
        assert_eq!(table.num_columns(), 0);
        assert_eq!(table.num_rows(), 0);
    }

    #[test]
    fn table_new() {
        let table = Table::new(Arc::new(Schema::default()), Vec::new(), HashMap::new())
            .expect("creating an empty `Table` should not fail");
        assert_eq!(table.num_columns(), 0);
        assert_eq!(table.num_rows(), 0);
    }

    #[test]
    fn column_new() {
        let column = Column::default();
        assert_eq!(column.len(), 0);
        assert_eq!(column.try_get::<UInt32ArrayType, u32>(0), Ok(None));

        let column = Column::default();
        assert_eq!(column.len(), 0);
        assert_eq!(column.try_get::<Utf8ArrayType, str>(0), Ok(None));
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
        let schema = Schema::new(vec![
            Field::new(DataType::Int64),
            Field::new(DataType::Utf8),
            Field::new(DataType::UInt32),
            Field::new(DataType::Float64),
            Field::new(DataType::Timestamp(TimeUnit::Second)),
            Field::new(DataType::UInt32),
            Field::new(DataType::Binary),
        ]);
        let c0_v: Vec<i64> = vec![1, 3, 3, 5, 2, 1, 3];
        let c1_v: Vec<_> = vec!["111a qwer", "b", "c", "d", "b", "111a qwer", "111a qwer"];
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
        let c4_v: Vec<i64> = vec![
            NaiveDate::from_ymd(2019, 9, 22)
                .and_hms(6, 10, 11)
                .timestamp(),
            NaiveDate::from_ymd(2019, 9, 22)
                .and_hms(6, 15, 11)
                .timestamp(),
            NaiveDate::from_ymd(2019, 9, 21)
                .and_hms(20, 10, 11)
                .timestamp(),
            NaiveDate::from_ymd(2019, 9, 21)
                .and_hms(20, 10, 11)
                .timestamp(),
            NaiveDate::from_ymd(2019, 9, 22)
                .and_hms(6, 45, 11)
                .timestamp(),
            NaiveDate::from_ymd(2019, 9, 21)
                .and_hms(8, 10, 11)
                .timestamp(),
            NaiveDate::from_ymd(2019, 9, 22)
                .and_hms(9, 10, 11)
                .timestamp(),
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

        let c0 = Column::try_from_slice::<Int64Type>(&c0_v).unwrap();
        let c1_a: Arc<dyn Array> = Arc::new(StringArray::try_from(c1_v.as_slice()).unwrap());
        let c1 = Column::from(c1_a);
        let c2 = Column::try_from_slice::<UInt32Type>(&c2_v).unwrap();
        let c3 = Column::try_from_slice::<Float64Type>(&c3_v).unwrap();
        let c4 = Column::try_from_slice::<Int64Type>(&c4_v).unwrap();
        let c5 = Column::try_from_slice::<UInt32Type>(&c5_v).unwrap();
        let c6_a: Arc<dyn Array> = Arc::new(BinaryArray::try_from(c6_v.as_slice()).unwrap());
        let c6 = Column::from(c6_a);
        let c_v: Vec<Column> = vec![c0, c1, c2, c3, c4, c5, c6];
        let table = Table::new(Arc::new(schema), c_v, HashMap::new()).expect("invalid columns");
        let column_types = Arc::new(vec![
            ColumnType::Int64,
            ColumnType::Utf8,
            ColumnType::IpAddr,
            ColumnType::Float64,
            ColumnType::DateTime,
            ColumnType::Enum,
            ColumnType::Binary,
        ]);
        let rows = vec![0_usize, 3, 1, 4, 2, 6, 5];
        let time_intervals = Arc::new(Vec::new());
        let numbers_of_top_n = Arc::new(Vec::new());
        let stat = table.get_statistics(
            &rows,
            &column_types,
            &reverse_enum_maps(&HashMap::new()),
            &time_intervals,
            &numbers_of_top_n,
        );

        assert_eq!(4, stat[0].n_largest_count.number_of_elements);
        assert_eq!(
            Element::Text("111a qwer".to_string()),
            *stat[1].n_largest_count.get_mode().unwrap()
        );
        assert_eq!(
            Element::IpAddr(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 3))),
            stat[2].n_largest_count.get_top_n().unwrap()[1].value
        );
        assert_eq!(3, stat[3].n_largest_count.number_of_elements);
        assert_eq!(
            Element::DateTime(NaiveDate::from_ymd(2019, 9, 22).and_hms(6, 0, 0)),
            stat[4].n_largest_count.get_top_n().unwrap()[0].value
        );
        assert_eq!(3, stat[5].n_largest_count.number_of_elements);
        assert_eq!(
            Element::Binary(b"111a qwer".to_vec()),
            *stat[6].n_largest_count.get_mode().unwrap()
        );

        let mut c5_map: HashMap<u32, String> = HashMap::new();
        c5_map.insert(1, "t1".to_string());
        c5_map.insert(2, "t2".to_string());
        c5_map.insert(7, "t3".to_string());
        let mut labels = HashMap::new();
        labels.insert(5, c5_map.into_iter().map(|(k, v)| (v, (k, 0))).collect());
        let stat = table.get_statistics(
            &rows,
            &column_types,
            &reverse_enum_maps(&labels),
            &time_intervals,
            &numbers_of_top_n,
        );

        assert_eq!(4, stat[0].n_largest_count.number_of_elements);
        assert_eq!(
            Element::Text("111a qwer".to_string()),
            *stat[1].n_largest_count.get_mode().unwrap()
        );
        assert_eq!(
            Element::IpAddr(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 3))),
            stat[2].n_largest_count.get_top_n().unwrap()[1].value
        );
        assert_eq!(3, stat[3].n_largest_count.number_of_elements);
        assert_eq!(
            Element::DateTime(NaiveDate::from_ymd(2019, 9, 22).and_hms(6, 0, 0)),
            stat[4].n_largest_count.get_top_n().unwrap()[0].value
        );
        assert_eq!(
            Element::Enum("t2".to_string()),
            *stat[5].n_largest_count.get_mode().unwrap()
        );
        assert_eq!(
            Element::Binary(b"111a qwer".to_vec()),
            *stat[6].n_largest_count.get_mode().unwrap()
        );
    }
}
