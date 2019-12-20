use crate::array::*;
use crate::datatypes::*;
use crate::memory::AllocationError;
use crate::{DataType, Schema};
use chrono::{NaiveDateTime, NaiveTime, Timelike};
use dashmap::DashMap;
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use statistical::*;
use std::collections::{HashMap, HashSet};
use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::hash::Hash;
use std::iter::{Flatten, Iterator};
use std::marker::PhantomData;
use std::net::{IpAddr, Ipv4Addr};
use std::ops::{Deref, Index};
use std::slice;
use std::sync::Arc;
use std::vec;

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
    Binary,
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
                DataType::Int32 => Column::new::<i32>(),
                DataType::Int64 | DataType::Timestamp(_) => Column::new::<i64>(),
                DataType::UInt8 => Column::new::<u8>(),
                DataType::UInt32 => Column::new::<u32>(),
                DataType::Float64 => Column::new::<f64>(),
                DataType::Utf8 => Column::new::<String>(),
                DataType::Binary => Column::new::<Vec<u8>>(),
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
    pub fn columns(&self) -> slice::Iter<Column> {
        self.columns.iter()
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
    ( $iter:expr, $len:expr, $d:expr, $t1:ty, $t2:expr ) => {
        let top_n_native: Vec<(&$t1, usize)> = count_sort($iter);
        $d.count = $len;
        $d.unique_count = top_n_native.len();
        let mut top_n: Vec<(DescriptionElement, usize)> = Vec::new();
        let top_n_num = if NUM_OF_TOP_N > top_n_native.len() {
            top_n_native.len()
        } else {
            NUM_OF_TOP_N
        };
        for (x, y) in &top_n_native[0..top_n_num] {
            top_n.push(($t2((*x).to_owned()), *y));
        }
        $d.mode = Some(top_n[0].0.clone());
        $d.top_n = Some(top_n);
    };
}

#[derive(Clone, Debug)]
pub struct Column {
    arrays: Vec<Arc<dyn Array>>,
    cumlen: Vec<usize>,
    len: usize,
}

impl Column {
    pub fn new<T>() -> Self
    where
        T: Send + Sync + 'static,
    {
        Self {
            arrays: Vec::new(),
            cumlen: vec![0],
            len: 0,
        }
    }

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
                let iter = self.view_iter::<Int64ArrayType, i64>(rows).unwrap();
                describe_min_max!(iter, desc, DescriptionElement::Int);
                let iter = self.view_iter::<Int64ArrayType, i64>(rows).unwrap();
                describe_top_n!(iter, rows.len(), desc, i64, DescriptionElement::Int);
                let iter = self.view_iter::<Int64ArrayType, i64>(rows).unwrap();
                #[allow(clippy::cast_precision_loss)] // 52-bit precision is good enough
                let f_values: Vec<f64> = iter.map(|v: &i64| *v as f64).collect();
                describe_mean_deviation!(f_values, i64, desc);
            }
            ColumnType::Float64 => {
                let iter = self.view_iter::<Float64ArrayType, f64>(rows).unwrap();
                describe_min_max!(iter, desc, DescriptionElement::Float);
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
                let iter = self.view_iter::<Float64ArrayType, f64>(rows).unwrap();
                let (rc, rt) = describe_top_n_f64(iter, *min, *max);
                desc.count = rows.len();
                desc.unique_count = rc;
                desc.mode = Some(rt[0].0.clone());
                desc.top_n = Some(rt);

                let iter = self.view_iter::<Float64ArrayType, f64>(rows).unwrap();
                let values = iter.cloned().collect::<Vec<_>>();
                describe_mean_deviation!(values, f64, desc);
            }
            ColumnType::Enum => {
                let iter = self.view_iter::<UInt32ArrayType, u32>(rows).unwrap();
                describe_top_n!(iter, self.len(), desc, u32, DescriptionElement::UInt);
            }
            ColumnType::Utf8 => {
                let iter = self.view_iter::<Utf8ArrayType, str>(rows).unwrap();
                describe_top_n!(iter, self.len(), desc, str, DescriptionElement::Text);
            }
            ColumnType::Binary => {
                let iter = self.view_iter::<BinaryArrayType, [u8]>(rows).unwrap();
                describe_top_n!(iter, self.len(), desc, [u8], DescriptionElement::Binary);
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
                    DescriptionElement::IpAddr
                );
            }
            ColumnType::DateTime => {
                let values = self
                    .view_iter::<Int64ArrayType, i64>(rows)
                    .unwrap()
                    .map(|v: &i64| {
                        let dt = NaiveDateTime::from_timestamp(*v, 0);
                        NaiveDateTime::new(dt.date(), NaiveTime::from_hms(dt.time().hour(), 0, 0))
                    })
                    .collect::<Vec<_>>();
                describe_top_n!(
                    values.iter(),
                    rows.len(),
                    desc,
                    NaiveDateTime,
                    DescriptionElement::DateTime
                );
            }
        }

        desc
    }
}

impl PartialEq for Column {
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
    ($name:ident, $array_ty:ty, $elem_ty:ty) => {
        pub struct $name<'a> {
            _marker: PhantomData<&'a u8>,
        }

        impl<'a> ArrayType for $name<'a> {
            type Array = $array_ty;
            type Elem = &'a $elem_ty;
        }
    };
}

make_array_type!(Int32ArrayType, primitive::Array<Int32Type>, i32);
make_array_type!(Int64ArrayType, primitive::Array<Int64Type>, i64);
make_array_type!(UInt8ArrayType, primitive::Array<UInt8Type>, u8);
make_array_type!(UInt32ArrayType, primitive::Array<UInt32Type>, u32);
make_array_type!(Float64ArrayType, primitive::Array<Float64Type>, f64);
make_array_type!(Utf8ArrayType, StringArray, str);
make_array_type!(BinaryArrayType, BinaryArray, [u8]);

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

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum DescriptionElement {
    Int(i64),
    UInt(u32),
    Enum(String), // describe() converts UInt -> Enum using enum maps. Without maps, by to_string().
    Float(f64),
    FloatRange(f64, f64),
    Text(String),
    Binary(Vec<u8>),
    IpAddr(IpAddr),
    DateTime(NaiveDateTime),
}

impl fmt::Display for DescriptionElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(x) => write!(f, "{}", x),
            Self::UInt(x) => write!(f, "{}", x),
            Self::Enum(x) | Self::Text(x) => write!(f, "{}", x),
            Self::Binary(x) => write!(f, "{:#?}", x),
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
            Self::Binary => "Binary",
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

fn describe_top_n_f64<I>(iter: I, min: f64, max: f64) -> (usize, Vec<(DescriptionElement, usize)>)
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
    fn table_new() {
        let table = Table::new(Vec::new(), HashMap::new());
        assert_eq!(table.num_columns(), 0);
        assert_eq!(table.num_rows(), 0);
    }

    #[test]
    fn table_try_from() {
        let table = Table::try_from(Vec::new());
        assert!(table.is_ok());
    }

    #[test]
    fn column_new() {
        let column = Column::new::<UInt32ArrayType>();
        assert_eq!(column.len(), 0);
        assert_eq!(column.try_get::<UInt32ArrayType, u32>(0), Ok(None));

        let column = Column::new::<Utf8ArrayType>();
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
        let table = Table::try_from(c_v).expect("invalid columns");
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
        assert_eq!(
            DescriptionElement::Binary(b"111a qwer".to_vec()),
            *ds[6].get_mode().unwrap()
        );

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
        assert_eq!(
            DescriptionElement::Binary(b"111a qwer".to_vec()),
            *ds[6].get_mode().unwrap()
        );
    }
}
