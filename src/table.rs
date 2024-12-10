use arrow::array::{
    Array, BinaryArray, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array,
    PrimitiveArray, PrimitiveBuilder, StringArray, UInt16Array, UInt32Array, UInt64Array,
    UInt8Array,
};
use arrow::datatypes::{ArrowPrimitiveType, DataType, Int64Type, Schema, TimeUnit};
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::iter::{Flatten, Iterator};
use std::marker::PhantomData;
use std::slice;
use std::sync::Arc;
use std::vec;
use strum_macros::EnumString;

use crate::stats::{
    convert_time_intervals, describe, n_largest_count, n_largest_count_datetime,
    n_largest_count_enum, n_largest_count_float64, ColumnStatistics, GroupCount, GroupElement,
    GroupElementCount,
};

type ReverseEnumMaps = HashMap<usize, HashMap<u64, Vec<String>>>;
/// The data type of a table column.
#[derive(Clone, Copy, Debug, Deserialize, EnumString, Eq, PartialEq, Serialize)]
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

impl From<ColumnType> for DataType {
    #[must_use]
    fn from(ct: ColumnType) -> Self {
        match ct {
            ColumnType::Int64 => Self::Int64,
            ColumnType::Float64 => Self::Float64,
            ColumnType::DateTime => Self::Timestamp(TimeUnit::Second, None),
            ColumnType::Enum | ColumnType::Utf8 => Self::Utf8,
            ColumnType::IpAddr => Self::UInt32,
            ColumnType::Binary => Self::Binary,
        }
    }
}

/// Structured data represented in a column-oriented form.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Table<I> {
    schema: Arc<Schema>,
    columns: Vec<Column>,
    event_ids: HashMap<I, usize>,
}

impl<I> Table<I>
where
    I: std::hash::Hash + PartialEq + Eq,
{
    /// Creates a new `Table` with the given `schema` and `columns`.
    ///
    /// # Errors
    ///
    /// Returns an error if `columns` have different lengths.
    pub fn new(
        schema: Arc<Schema>,
        columns: Vec<Column>,
        event_ids: HashMap<I, usize>,
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
    pub fn columns(&self) -> slice::Iter<Column> {
        self.columns.iter()
    }

    /// Returns an immutable reference of a specific column
    #[must_use]
    pub fn column(&self, i: usize) -> Option<&Column> {
        self.columns.get(i)
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

    /// Returns the schema of the table.
    #[must_use]
    pub fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    /// # Panics
    ///
    /// Panics if time intervals or number of top n is not defined.
    #[must_use]
    pub fn statistics(
        &self,
        rows: &[usize],
        column_types: &Arc<Vec<ColumnType>>,
        r_enum_maps: &ReverseEnumMaps,
        time_intervals: &Arc<Vec<u32>>,
        numbers_of_top_n: &Arc<Vec<u32>>,
        precision: i32,
    ) -> Vec<ColumnStatistics> {
        self.columns
            .iter()
            .enumerate()
            .map(|(index, column)| {
                let description = describe(column, rows, column_types[index]);
                let n_largest_count = if let ColumnType::Enum = column_types[index] {
                    n_largest_count_enum(
                        column,
                        rows,
                        r_enum_maps.get(&index).unwrap_or(&HashMap::new()),
                        *numbers_of_top_n
                            .get(index)
                            .expect("top N number for each column should exist."),
                    )
                } else if let ColumnType::DateTime = column_types[index] {
                    let mut cn: usize = 0;
                    for i in 0..index {
                        if let ColumnType::DateTime = column_types[i] {
                            cn += 1;
                        }
                    }
                    n_largest_count_datetime(
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
                    n_largest_count_float64(
                        column,
                        rows,
                        *numbers_of_top_n
                            .get(index)
                            .expect("top N number for each column should exist."),
                        precision,
                    )
                } else {
                    n_largest_count(
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

    // count means including only positive values. Implement other functions like sum_group_by, mean_group_by, etc. later.
    /// # Panics
    ///
    /// Panics if columns are not defined.
    #[must_use]
    pub fn count_group_by(
        &self,
        rows: &[usize],
        column_types: &Arc<Vec<ColumnType>>,
        by_column: usize,
        by_interval: Option<u32>,
        count_columns: &Arc<Vec<usize>>,
    ) -> Vec<GroupCount> {
        let column_type = if let Some(column_type) = column_types.get(by_column) {
            *column_type
        } else {
            return Vec::new();
        };

        let rows_interval: Vec<GroupElement> = match column_type {
            ColumnType::DateTime => {
                if let Some(by_interval) = by_interval {
                    convert_time_intervals(
                        self.columns
                            .get(by_column)
                            .expect("time column should exist"),
                        rows,
                        by_interval,
                    )
                    .iter()
                    .map(|e| GroupElement::DateTime(*e))
                    .collect()
                } else {
                    return Vec::new();
                }
            }
            _ => return Vec::new(), // TODO: implement other types
        };

        count_columns
            .iter()
            .filter_map(|&count_index| {
                let column = self.columns.get(count_index)?;

                let mut element_count: HashMap<GroupElement, usize> = HashMap::new();
                if by_column == count_index {
                    for r in &rows_interval {
                        *element_count.entry(r.clone()).or_insert(0) += 1; // count just rows
                    }
                } else if let ColumnType::Int64 = column_types[count_index] {
                    let counts = column
                        .primitive_iter::<Int64Type>(rows)
                        .expect("expecting Int64Type only")
                        .map(|v| v.to_usize().unwrap_or(0)) // if count is negative, then 0
                        .collect::<Vec<_>>();

                    for (index, r) in rows_interval.iter().enumerate() {
                        *element_count.entry(r.clone()).or_insert(0) += counts[index];
                        // count column values
                    }
                }

                if element_count.is_empty() {
                    None
                } else {
                    let mut series: Vec<GroupElementCount> = element_count
                        .iter()
                        .map(|(value, &count)| GroupElementCount {
                            value: value.clone(),
                            count,
                        })
                        .collect();

                    series
                        .sort_by(|a, b| a.value.partial_cmp(&b.value).expect("always comparable"));

                    let count_index = if by_column == count_index {
                        None
                    } else {
                        Some(count_index)
                    };
                    Some(GroupCount {
                        count_index,
                        series,
                    })
                }
            })
            .collect()
    }

    #[must_use]
    pub fn event_index(&self, eventid: &I) -> Option<&usize> {
        self.event_ids.get(eventid)
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
    /// Returns an error if array operation failed.
    pub fn try_from_slice<T>(slice: &[T::Native]) -> arrow::error::Result<Self>
    where
        T: ArrowPrimitiveType,
    {
        let mut builder = PrimitiveBuilder::<T>::new();
        for s in slice {
            builder.append_value(*s);
        }
        let array: Arc<dyn Array> = Arc::new(builder.finish());
        Ok(array.into())
    }

    fn len(&self) -> usize {
        self.len
    }

    /// Return the value specified by the index as type T
    ///
    /// # Errors
    ///
    /// Returns an error if it's fail to convert the value to the specified type
    pub fn primitive_try_get<T>(&self, index: usize) -> Result<Option<T::Native>, TypeError>
    where
        T: ArrowPrimitiveType,
    {
        if index >= self.len() {
            return Ok(None);
        }
        let (array_index, inner_index) = match self.cumlen.binary_search(&index) {
            Ok(i) => (i, 0),
            Err(i) => (i - 1, index - self.cumlen[i - 1]),
        };
        let Some(typed_arr) = self.arrays[array_index]
            .as_any()
            .downcast_ref::<PrimitiveArray<T>>()
        else {
            return Err(TypeError());
        };
        Ok(Some(typed_arr.value(inner_index)))
    }

    /// Return the value specified by the index as byte array
    ///
    /// # Errors
    ///
    /// Returns an error if it's fail to convert the value to byte array
    pub fn binary_try_get(&self, index: usize) -> Result<Option<&[u8]>, TypeError> {
        if index >= self.len() {
            return Ok(None);
        }
        let (array_index, inner_index) = match self.cumlen.binary_search(&index) {
            Ok(i) => (i, 0),
            Err(i) => (i - 1, index - self.cumlen[i - 1]),
        };
        let Some(typed_arr) = self.arrays[array_index]
            .as_any()
            .downcast_ref::<BinaryArray>()
        else {
            return Err(TypeError());
        };
        Ok(Some(typed_arr.value(inner_index)))
    }

    /// Return the value specified by the index as string
    ///
    /// # Errors
    ///
    /// Returns an error if it's fail to convert the value to string
    pub fn string_try_get(&self, index: usize) -> Result<Option<&str>, TypeError> {
        if index >= self.len() {
            return Ok(None);
        }
        let (array_index, inner_index) = match self.cumlen.binary_search(&index) {
            Ok(i) => (i, 0),
            Err(i) => (i - 1, index - self.cumlen[i - 1]),
        };
        let Some(typed_arr) = self.arrays[array_index]
            .as_any()
            .downcast_ref::<StringArray>()
        else {
            return Err(TypeError());
        };
        Ok(Some(typed_arr.value(inner_index)))
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
    #[allow(clippy::iter_not_returning_iterator)]
    // TODO: resolve this clippy warning
    pub fn iter<'a, T>(&'a self) -> Result<Flatten<vec::IntoIter<&'a T>>, TypeError>
    where
        T: Array + 'static,
        &'a T: IntoIterator,
    {
        let mut arrays: Vec<&T> = Vec::with_capacity(self.arrays.len());
        for arr in &self.arrays {
            let Some(typed_arr) = arr.as_any().downcast_ref::<T>() else {
                return Err(TypeError());
            };
            arrays.push(typed_arr);
        }
        Ok(arrays.into_iter().flatten())
    }

    /// Creates an iterator iterating over a subset of the cells in this
    /// `Column` of primitive type, designated by `selected`.
    ///
    /// # Errors
    ///
    /// Returns an error if the type parameter does not match with the type of
    /// this `Column`.
    pub fn primitive_iter<'a, 'b, T>(
        &'a self,
        selected: &'b [usize],
    ) -> Result<PrimitiveIter<'a, 'b, T>, TypeError>
    where
        T: ArrowPrimitiveType,
    {
        Ok(PrimitiveIter::new(self, selected.iter()))
    }

    /// Creates an iterator iterating over a subset of the cells in this
    /// `Column` of binaries, designated by `selected`.
    ///
    /// # Errors
    ///
    /// Returns an error if the type parameter does not match with the type of
    /// this `Column`.
    pub fn binary_iter<'a, 'b>(
        &'a self,
        selected: &'b [usize],
    ) -> Result<BinaryIter<'a, 'b>, TypeError> {
        Ok(BinaryIter::new(self, selected.iter()))
    }

    /// Creates an iterator iterating over a subset of the cells in this
    /// `Column` of strings, designated by `selected`.
    ///
    /// # Errors
    ///
    /// Returns an error if the type parameter does not match with the type of
    /// this `Column`.
    pub fn string_iter<'a, 'b>(
        &'a self,
        selected: &'b [usize],
    ) -> Result<StringIter<'a, 'b>, TypeError> {
        Ok(StringIter::new(self, selected.iter()))
    }
}

impl PartialEq for Column {
    #[must_use]
    fn eq(&self, other: &Self) -> bool {
        let data_type = match (self.arrays.first(), other.arrays.first()) {
            (Some(x_arr), Some(y_arr)) => {
                if x_arr.to_data().data_type() == y_arr.to_data().data_type() {
                    x_arr.to_data().data_type().clone()
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
            DataType::Int8 => self
                .iter::<Int8Array>()
                .expect("invalid array")
                .zip(other.iter::<Int8Array>().expect("invalid array"))
                .all(|(x, y)| x == y),
            DataType::Int16 => self
                .iter::<Int16Array>()
                .expect("invalid array")
                .zip(other.iter::<Int16Array>().expect("invalid array"))
                .all(|(x, y)| x == y),
            DataType::Int32 => self
                .iter::<Int32Array>()
                .expect("invalid array")
                .zip(other.iter::<Int32Array>().expect("invalid array"))
                .all(|(x, y)| x == y),
            DataType::Int64 | DataType::Timestamp(_, _) => self
                .iter::<Int64Array>()
                .expect("invalid array")
                .zip(other.iter::<Int64Array>().expect("invalid array"))
                .all(|(x, y)| x == y),
            DataType::UInt8 => self
                .iter::<UInt8Array>()
                .expect("invalid array")
                .zip(other.iter::<UInt8Array>().expect("invalid array"))
                .all(|(x, y)| x == y),
            DataType::UInt16 => self
                .iter::<UInt16Array>()
                .expect("invalid array")
                .zip(other.iter::<UInt16Array>().expect("invalid array"))
                .all(|(x, y)| x == y),
            DataType::UInt32 => self
                .iter::<UInt32Array>()
                .expect("invalid array")
                .zip(other.iter::<UInt32Array>().expect("invalid array"))
                .all(|(x, y)| x == y),
            DataType::UInt64 => self
                .iter::<UInt64Array>()
                .expect("invalid array")
                .zip(other.iter::<UInt64Array>().expect("invalid array"))
                .all(|(x, y)| x == y),
            DataType::Float32 => self
                .iter::<Float32Array>()
                .expect("invalid array")
                .zip(other.iter::<Float32Array>().expect("invalid array"))
                .all(|(x, y)| x == y),
            DataType::Float64 => self
                .iter::<Float64Array>()
                .expect("invalid array")
                .zip(other.iter::<Float64Array>().expect("invalid array"))
                .all(|(x, y)| x == y),
            DataType::Utf8 => self
                .iter::<StringArray>()
                .expect("invalid array")
                .zip(other.iter::<StringArray>().expect("invalid array"))
                .all(|(x, y)| x == y),
            DataType::Binary => self
                .iter::<BinaryArray>()
                .expect("invalid array")
                .zip(other.iter::<BinaryArray>().expect("invalid array"))
                .all(|(x, y)| x == y),
            _ => unimplemented!(),
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

#[derive(Debug, Eq, PartialEq)]
pub struct TypeError();

pub struct PrimitiveIter<'a, 'b, T: ArrowPrimitiveType> {
    column: &'a Column,
    selected: slice::Iter<'b, usize>,
    _t_marker: PhantomData<T>,
}

impl<'a, 'b, T> PrimitiveIter<'a, 'b, T>
where
    T: ArrowPrimitiveType,
{
    fn new(column: &'a Column, selected: slice::Iter<'b, usize>) -> Self {
        Self {
            column,
            selected,
            _t_marker: PhantomData,
        }
    }
}

impl<'a, 'b, T> Iterator for PrimitiveIter<'a, 'b, T>
where
    T: ArrowPrimitiveType,
{
    type Item = T::Native;

    fn next(&mut self) -> Option<Self::Item> {
        let selected = self.selected.next()?;
        if let Ok(elem) = self.column.primitive_try_get::<T>(*selected) {
            elem
        } else {
            None
        }
    }
}

pub struct BinaryIter<'a, 'b> {
    column: &'a Column,
    selected: slice::Iter<'b, usize>,
}

impl<'a, 'b> BinaryIter<'a, 'b> {
    fn new(column: &'a Column, selected: slice::Iter<'b, usize>) -> Self {
        Self { column, selected }
    }
}

impl<'a, 'b> Iterator for BinaryIter<'a, 'b> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        let selected = self.selected.next()?;
        if let Ok(elem) = self.column.binary_try_get(*selected) {
            elem
        } else {
            None
        }
    }
}

pub struct StringIter<'a, 'b> {
    column: &'a Column,
    selected: slice::Iter<'b, usize>,
}

impl<'a, 'b> StringIter<'a, 'b> {
    fn new(column: &'a Column, selected: slice::Iter<'b, usize>) -> Self {
        Self { column, selected }
    }
}

impl<'a, 'b> Iterator for StringIter<'a, 'b> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        let selected = self.selected.next()?;
        if let Ok(elem) = self.column.string_try_get(*selected) {
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
    use ahash::AHasher;
    use arrow::datatypes::{Field, Float64Type, UInt32Type, UInt64Type};
    use chrono::NaiveDate;
    use std::hash::{Hash, Hasher};
    use std::net::{IpAddr, Ipv4Addr};

    fn hash(seq: &str) -> u64 {
        let mut hasher = AHasher::default();
        seq.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn table_new() {
        let table = Table::<usize>::new(Arc::new(Schema::empty()), Vec::new(), HashMap::new())
            .expect("creating an empty `Table` should not fail");
        assert_eq!(table.num_columns(), 0);
        assert_eq!(table.num_rows(), 0);
    }

    #[test]
    fn column_new() {
        let column = Column::default();
        assert_eq!(column.len(), 0);
        assert_eq!(column.primitive_try_get::<UInt32Type>(0), Ok(None));

        let column = Column::default();
        assert_eq!(column.len(), 0);
        assert_eq!(column.string_try_get(0), Ok(None));
    }

    #[test]
    fn check_schema() {
        let schema = Schema::new(vec![
            Field::new("ts", DataType::Timestamp(TimeUnit::Second, None), false),
            Field::new("len", DataType::Int64, false),
            Field::new("last_time", DataType::Int64, false),
        ]);
        let table = Table::<usize>::new(Arc::new(schema), Vec::new(), HashMap::new())
            .expect("creating an empty `Table` should not fail");

        let ret_schema = table.schema();
        assert_eq!(ret_schema.all_fields().len(), 3);
        let ret_field_ts = ret_schema.field_with_name("ts");
        assert!(ret_field_ts.is_ok());
        if let Ok(ts) = ret_field_ts {
            assert_eq!(ts.name(), "ts");
        }
    }

    #[test]
    fn count_group_by_test() {
        let schema = Schema::new(vec![
            Field::new("", DataType::Timestamp(TimeUnit::Second, None), false),
            Field::new("", DataType::Int64, false),
            Field::new("", DataType::Int64, false),
        ]);
        let c0_v: Vec<i64> = vec![
            NaiveDate::from_ymd_opt(2020, 1, 1)
                .unwrap()
                .and_hms_opt(0, 0, 10)
                .unwrap()
                .and_utc()
                .timestamp_nanos_opt()
                .unwrap(),
            NaiveDate::from_ymd_opt(2020, 1, 1)
                .unwrap()
                .and_hms_opt(0, 0, 13)
                .unwrap()
                .and_utc()
                .timestamp_nanos_opt()
                .unwrap(),
            NaiveDate::from_ymd_opt(2020, 1, 1)
                .unwrap()
                .and_hms_opt(0, 0, 15)
                .unwrap()
                .and_utc()
                .timestamp_nanos_opt()
                .unwrap(),
            NaiveDate::from_ymd_opt(2020, 1, 1)
                .unwrap()
                .and_hms_opt(0, 0, 22)
                .unwrap()
                .and_utc()
                .timestamp_nanos_opt()
                .unwrap(),
            NaiveDate::from_ymd_opt(2020, 1, 1)
                .unwrap()
                .and_hms_opt(0, 0, 22)
                .unwrap()
                .and_utc()
                .timestamp_nanos_opt()
                .unwrap(),
            NaiveDate::from_ymd_opt(2020, 1, 1)
                .unwrap()
                .and_hms_opt(0, 0, 31)
                .unwrap()
                .and_utc()
                .timestamp_nanos_opt()
                .unwrap(),
            NaiveDate::from_ymd_opt(2020, 1, 1)
                .unwrap()
                .and_hms_opt(0, 0, 33)
                .unwrap()
                .and_utc()
                .timestamp_nanos_opt()
                .unwrap(),
            NaiveDate::from_ymd_opt(2020, 1, 1)
                .unwrap()
                .and_hms_opt(0, 1, 1)
                .unwrap()
                .and_utc()
                .timestamp_nanos_opt()
                .unwrap(),
        ];
        let c1_v: Vec<i64> = vec![1, 32, 3, 5, 2, 1, 3, 24];
        let c2_v: Vec<i64> = vec![2, 33, 4, 6, 3, 2, 4, 25];
        let c0 = Column::try_from_slice::<Int64Type>(&c0_v).unwrap();
        let c1 = Column::try_from_slice::<Int64Type>(&c1_v).unwrap();
        let c2 = Column::try_from_slice::<Int64Type>(&c2_v).unwrap();
        let c_v: Vec<Column> = vec![c0, c1, c2];
        let table =
            Table::<i64>::new(Arc::new(schema), c_v, HashMap::new()).expect("invalid columns");
        let column_types = Arc::new(vec![
            ColumnType::DateTime,
            ColumnType::Int64,
            ColumnType::Int64,
        ]);
        let rows = vec![0_usize, 3, 1, 4, 2, 6, 5, 7];
        let count_columns = vec![0, 1, 2];
        let group_count =
            table.count_group_by(&rows, &column_types, 0, Some(30), &Arc::new(count_columns));
        assert_eq!(None, group_count[0].count_index);
        assert_eq!(Some(1), group_count[1].count_index);
        assert_eq!(Some(2), group_count[2].count_index);
        assert_eq!(5_usize, group_count[0].series[0].count);
        assert_eq!(43_usize, group_count[1].series[0].count);
        assert_eq!(48_usize, group_count[2].series[0].count);
    }

    #[test]
    fn description_test() {
        use crate::Element;
        let schema = Schema::new(vec![
            Field::new("", DataType::Int64, false),
            Field::new("", DataType::Utf8, false),
            Field::new("", DataType::UInt32, false),
            Field::new("", DataType::Float64, false),
            Field::new("", DataType::Timestamp(TimeUnit::Second, None), false),
            Field::new("", DataType::UInt64, false),
            Field::new("", DataType::Binary, false),
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
        let c3_v: Vec<f64> = vec![2.2, 2.203, 2.8, 5.30123, 7.0, 10320.811, 5.3009];
        let c4_v: Vec<i64> = vec![
            NaiveDate::from_ymd_opt(2019, 9, 22)
                .unwrap()
                .and_hms_opt(6, 10, 11)
                .unwrap()
                .and_utc()
                .timestamp_nanos_opt()
                .unwrap(),
            NaiveDate::from_ymd_opt(2019, 9, 22)
                .unwrap()
                .and_hms_opt(6, 15, 11)
                .unwrap()
                .and_utc()
                .timestamp_nanos_opt()
                .unwrap(),
            NaiveDate::from_ymd_opt(2019, 9, 21)
                .unwrap()
                .and_hms_opt(20, 10, 11)
                .unwrap()
                .and_utc()
                .timestamp_nanos_opt()
                .unwrap(),
            NaiveDate::from_ymd_opt(2019, 9, 21)
                .unwrap()
                .and_hms_opt(20, 10, 11)
                .unwrap()
                .and_utc()
                .timestamp_nanos_opt()
                .unwrap(),
            NaiveDate::from_ymd_opt(2019, 9, 22)
                .unwrap()
                .and_hms_opt(6, 45, 11)
                .unwrap()
                .and_utc()
                .timestamp_nanos_opt()
                .unwrap(),
            NaiveDate::from_ymd_opt(2019, 9, 21)
                .unwrap()
                .and_hms_opt(8, 10, 11)
                .unwrap()
                .and_utc()
                .timestamp_nanos_opt()
                .unwrap(),
            NaiveDate::from_ymd_opt(2019, 9, 22)
                .unwrap()
                .and_hms_opt(9, 10, 11)
                .unwrap()
                .and_utc()
                .timestamp_nanos_opt()
                .unwrap(),
        ];
        let tester = vec!["t1".to_string(), "t2".to_string(), "t3".to_string()];
        let sid = tester.iter().map(|s| hash(s)).collect::<Vec<_>>();
        let c5_v: Vec<u64> = vec![sid[0], sid[1], sid[1], sid[1], sid[1], sid[1], sid[2]];
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
        let c1_a: Arc<dyn Array> = Arc::new(StringArray::from(c1_v));
        let c1 = Column::from(c1_a);
        let c2 = Column::try_from_slice::<UInt32Type>(&c2_v).unwrap();
        let c3 = Column::try_from_slice::<Float64Type>(&c3_v).unwrap();
        let c4 = Column::try_from_slice::<Int64Type>(&c4_v).unwrap();
        let c5 = Column::try_from_slice::<UInt64Type>(&c5_v).unwrap();
        let c6_a: Arc<dyn Array> = Arc::new(BinaryArray::from(c6_v));
        let c6 = Column::from(c6_a);
        let c_v: Vec<Column> = vec![c0, c1, c2, c3, c4, c5, c6];
        let table =
            Table::<i64>::new(Arc::new(schema), c_v, HashMap::new()).expect("invalid columns");
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
        let stat = table.statistics(
            &rows,
            &column_types,
            &HashMap::new(),
            &time_intervals,
            &numbers_of_top_n,
            2,
        );

        assert_eq!(4, stat[0].n_largest_count.number_of_elements());
        assert_eq!(
            Element::Text("111a qwer".to_string()),
            *stat[1].n_largest_count.mode().unwrap()
        );
        assert_eq!(
            Element::IpAddr(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 3))),
            stat[2].n_largest_count.top_n()[1].value
        );
        assert_eq!(5, stat[3].n_largest_count.number_of_elements());
        assert_eq!(
            Element::DateTime(
                NaiveDate::from_ymd_opt(2019, 9, 22)
                    .unwrap()
                    .and_hms_opt(6, 0, 0)
                    .unwrap()
            ),
            stat[4].n_largest_count.top_n()[0].value
        );
        assert_eq!(3, stat[5].n_largest_count.number_of_elements());
        assert_eq!(
            Element::Binary(b"111a qwer".to_vec()),
            *stat[6].n_largest_count.mode().unwrap()
        );

        let c5_r_map: ReverseEnumMaps = vec![(
            5,
            sid.iter()
                .zip(tester.iter())
                .map(|(id, s)| (*id, vec![s.to_string()]))
                .collect(),
        )]
        .into_iter()
        .collect();
        let stat = table.statistics(
            &rows,
            &column_types,
            &c5_r_map,
            &time_intervals,
            &numbers_of_top_n,
            2,
        );

        assert_eq!(4, stat[0].n_largest_count.number_of_elements());
        assert_eq!(
            Element::Text("111a qwer".to_string()),
            *stat[1].n_largest_count.mode().unwrap()
        );
        assert_eq!(
            Element::IpAddr(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 3))),
            stat[2].n_largest_count.top_n()[1].value
        );
        assert_eq!(5, stat[3].n_largest_count.number_of_elements());
        assert_eq!(
            Element::DateTime(
                NaiveDate::from_ymd_opt(2019, 9, 22)
                    .unwrap()
                    .and_hms_opt(6, 0, 0)
                    .unwrap()
            ),
            stat[4].n_largest_count.top_n()[0].value
        );
        assert_eq!(
            Element::Enum("t2".to_string()),
            *stat[5].n_largest_count.mode().unwrap()
        );
        assert_eq!(
            Element::Binary(b"111a qwer".to_vec()),
            *stat[6].n_largest_count.mode().unwrap()
        );
    }
}
