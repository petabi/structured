use chrono::{NaiveDate, NaiveDateTime, NaiveTime};
use csv::ByteRecord;
use itertools::izip;
use std::any::Any;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::net::{IpAddr, Ipv4Addr};
use std::slice::Iter;

use crate::{DataType, Schema};

#[derive(Debug, Default)]
pub struct Table {
    columns: Vec<Column>,
}

impl Table {
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
                DataType::DateTime(_) => Column::new::<NaiveDateTime>(),
            })
            .collect();
        Self { columns }
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

    pub fn push(
        &mut self,
        schema: &Schema,
        values: &ByteRecord,
        labels: Option<&HashMap<usize, HashMap<String, u32>>>,
    ) -> Result<(), &'static str> {
        if self.columns.len() != schema.fields().len() {
            return Err("# of fields in the format is different from # of columns in the table");
        }
        if self.columns.len() != values.len() {
            return Err("# of values is different from # of columns in the table");
        }
        for (i, (col, fmt, val)) in izip!(
            self.columns.iter_mut(),
            schema.fields().iter(),
            values.iter()
        )
        .enumerate()
        {
            let val: String = val.iter().map(|&c| c as char).collect();
            if let Some(v) = col.values_mut::<i64>() {
                v.push(val.parse::<i64>().unwrap_or_default());
            } else if let Some(v) = col.values_mut::<f64>() {
                v.push(val.parse::<f64>().unwrap_or_default());
            } else if let Some(v) = col.values_mut::<u32>() {
                let enum_value = labels.as_ref().map_or(0_u32, |label_map| {
                    label_map
                        .get(&i)
                        .unwrap()
                        .get(&val.to_string())
                        .map_or(0, |val| *val)
                });
                v.push(enum_value);
            } else if let Some(v) = col.values_mut::<String>() {
                v.push(val);
            } else if let Some(v) = col.values_mut::<IpAddr>() {
                v.push(
                    val.parse::<IpAddr>()
                        .unwrap_or_else(|_| IpAddr::V4(Ipv4Addr::new(255, 255, 255, 255))),
                );
            } else if let Some(v) = col.values_mut::<NaiveDateTime>() {
                let fmt = if let DataType::DateTime(fmt) = &fmt.data_type() {
                    fmt
                } else {
                    return Err("column type mismatch");
                };
                v.push(
                    NaiveDateTime::parse_from_str(&val, fmt).unwrap_or_else(|_| {
                        NaiveDateTime::new(
                            NaiveDate::from_ymd(1, 1, 1),
                            NaiveTime::from_hms(0, 0, 0),
                        )
                    }),
                );
            } else {
                return Err("unknown column type");
            }
        }
        Ok(())
    }

    /// Returns an `Interator` for columns.
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
}

impl TryFrom<Vec<Column>> for Table {
    type Error = &'static str;

    fn try_from(columns: Vec<Column>) -> Result<Self, Self::Error> {
        let len = if let Some(col) = columns.first() {
            col.len()
        } else {
            return Ok(Self { columns });
        };
        if columns.iter().skip(1).all(|e| e.len() == len) {
            Ok(Self { columns })
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
