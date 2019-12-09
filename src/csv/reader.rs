use chrono::format::ParseError as TimeParseError;
use chrono::NaiveDateTime;
use std::fmt;
use std::sync::Arc;

pub struct ParseError {
    inner: Box<dyn std::error::Error>,
}

impl fmt::Debug for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "parse error: {}", self.inner)
    }
}

impl From<std::net::AddrParseError> for ParseError {
    fn from(error: std::net::AddrParseError) -> Self {
        Self {
            inner: Box::new(error),
        }
    }
}

impl From<std::num::ParseIntError> for ParseError {
    fn from(error: std::num::ParseIntError) -> Self {
        Self {
            inner: Box::new(error),
        }
    }
}

impl From<std::str::Utf8Error> for ParseError {
    fn from(error: std::str::Utf8Error) -> Self {
        Self {
            inner: Box::new(error),
        }
    }
}

pub type UInt32Parser = dyn Fn(&[u8]) -> Result<u32, ParseError> + Send + Sync;
pub type DateTimeParser = dyn Fn(&[u8]) -> Result<NaiveDateTime, TimeParseError> + Send + Sync;

#[derive(Clone)]
pub enum FieldParser {
    Int64,
    UInt32(Arc<UInt32Parser>),
    Float64,
    Utf8,
    DateTime(Arc<DateTimeParser>),
    Dict,
}

impl FieldParser {
    pub fn default_uint32() -> Self {
        Self::UInt32(Arc::new(parse_uint32))
    }

    pub fn uint32_with_parser<P>(parser: P) -> Self
    where
        P: Fn(&[u8]) -> Result<u32, ParseError> + Send + Sync + 'static,
    {
        Self::UInt32(Arc::new(parser))
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
            Self::UInt32(_) => write!(f, "UInt32"),
            Self::Float64 => write!(f, "Float64"),
            Self::Utf8 => write!(f, "Utf8"),
            Self::DateTime(_) => write!(f, "DateTime(<Fn>)"),
            Self::Dict => write!(f, "Dict"),
        }
    }
}

fn parse_uint32(v: &[u8]) -> Result<u32, ParseError> {
    std::str::from_utf8(v)?.parse::<u32>().map_err(Into::into)
}
