use chrono::format::ParseError as TimeParseError;
use chrono::NaiveDateTime;
use std::fmt;
use std::net::{AddrParseError, IpAddr};
use std::sync::Arc;

pub type IpAddrParser = dyn Fn(&[u8]) -> Result<IpAddr, AddrParseError> + Send + Sync;
pub type DateTimeParser = dyn Fn(&[u8]) -> Result<NaiveDateTime, TimeParseError> + Send + Sync;

#[derive(Clone)]
pub enum FieldParser {
    Int64,
    UInt32,
    Float64,
    Utf8,
    IpAddr(Arc<IpAddrParser>),
    DateTime(Arc<DateTimeParser>),
    Dict,
}

impl FieldParser {
    pub fn new_ipaddr<P>(parser: P) -> Self
    where
        P: Fn(&[u8]) -> Result<IpAddr, AddrParseError> + Send + Sync + 'static,
    {
        Self::IpAddr(Arc::new(parser))
    }

    pub fn new_datetime<P>(parser: P) -> Self
    where
        P: Fn(&[u8]) -> Result<NaiveDateTime, TimeParseError> + Send + Sync + 'static,
    {
        Self::DateTime(Arc::new(parser))
    }
}

impl<'a> fmt::Debug for FieldParser {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int64 => write!(f, "Int64"),
            Self::UInt32 => write!(f, "UInt32"),
            Self::Float64 => write!(f, "Float64"),
            Self::Utf8 => write!(f, "Utf8"),
            Self::IpAddr(_) => write!(f, "IpAddr(<Fn>)"),
            Self::DateTime(_) => write!(f, "DateTime(<Fn>)"),
            Self::Dict => write!(f, "Dict"),
        }
    }
}
