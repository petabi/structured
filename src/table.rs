use crate::array::*;
use crate::datatypes::*;
use crate::memory::AllocationError;
use crate::{DataType, Schema};
use dashmap::DashMap;
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::iter::{Flatten, Iterator};
use std::marker::PhantomData;
use std::ops::Index;
use std::slice;
use std::sync::Arc;
use std::vec;
use strum_macros::EnumString;

use crate::stats::{
    describe, get_n_largest_count, get_n_largest_count_datetime, get_n_largest_count_enum,
    get_n_largest_count_float64, ColumnStatistics, Element, NLargestCount,
};

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
                let description = describe(column, rows, column_types[index]);
                let n_largest_count = if let ColumnType::Enum = column_types[index] {
                    get_n_largest_count_enum(
                        column,
                        rows,
                        r_enum_maps.get(&index).unwrap_or(&Arc::new(HashMap::new())),
                        *numbers_of_top_n
                            .get(index)
                            .expect("top N number for each column should exist."),
                    )
                } else if let ColumnType::DateTime = column_types[index] {
                    let mut cn = 0_usize;
                    for i in 0..index {
                        if let ColumnType::DateTime = column_types[i] {
                            cn += 1;
                        }
                    }
                    get_n_largest_count_datetime(
                        column,
                        rows,
                        *time_intervals
                            .get(cn)
                            .expect("time intervals should exist."),
                        *numbers_of_top_n
                            .get(index)
                            .expect("top N number for each column should exist."),
                    )
                } else if let ColumnType::Float64 = column_types[index] {
                    if let (Some(Element::Float(min)), Some(Element::Float(max))) =
                        (description.get_min(), description.get_max())
                    {
                        get_n_largest_count_float64(
                            column,
                            rows,
                            *numbers_of_top_n
                                .get(index)
                                .expect("top N number for each column should exist."),
                            *min,
                            *max,
                        )
                    } else {
                        NLargestCount::default()
                    }
                } else {
                    get_n_largest_count(
                        column,
                        rows,
                        column_types[index],
                        *numbers_of_top_n
                            .get(index)
                            .expect("top N number for each column should exist."),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Column;
    use chrono::NaiveDate;
    use std::convert::TryFrom;
    use std::net::{IpAddr, Ipv4Addr};

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
        let time_intervals = Arc::new(vec![3600]);
        let numbers_of_top_n = Arc::new(vec![10; 7]);
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
