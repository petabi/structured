use crate::{Column, DataType, Schema};
use chrono::{NaiveDate, NaiveDateTime, NaiveTime};
use csv::{ByteRecord, ByteRecordIter};
use dashmap::DashMap;
use num_traits::ToPrimitive;
use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr};
use std::sync::Arc;

type ConcurrentEnumMap = Arc<DashMap<usize, Arc<DashMap<String, (u32, usize)>>>>;

pub fn records_to_columns<S: ::std::hash::BuildHasher>(
    values: &[ByteRecord],
    schema: &Schema,
    labels: &ConcurrentEnumMap,
    formats: &HashMap<usize, String, S>,
) -> Vec<Column> {
    let mut records: Vec<ByteRecordIter> = values.iter().map(ByteRecord::iter).collect();
    schema
        .fields()
        .iter()
        .enumerate()
        .map(|(fid, field)| match field.data_type() {
            DataType::Int => Column::with_data(records.iter_mut().fold(vec![], |mut col, v| {
                let val: String = v.next().unwrap().iter().map(|&c| c as char).collect();
                col.push(val.parse::<i64>().unwrap_or_default());
                col
            })),
            DataType::Float => Column::with_data(records.iter_mut().fold(vec![], |mut col, v| {
                let val: String = v.next().unwrap().iter().map(|&c| c as char).collect();
                col.push(val.parse::<f64>().unwrap_or_default());
                col
            })),
            DataType::Str => Column::with_data(records.iter_mut().fold(vec![], |mut col, v| {
                let val: String = v.next().unwrap().iter().map(|&c| c as char).collect();
                col.push(val);
                col
            })),
            DataType::Enum => Column::with_data(records.iter_mut().fold(vec![], |mut col, v| {
                let val: String = v.next().unwrap().iter().map(|&c| c as char).collect();
                let enum_value = if let Some(map) = labels.get(&fid) {
                    let enum_value = map
                        .get_or_insert(
                            &val,
                            (
                                (map.len() + 1).to_u32().unwrap_or(u32::max_value()),
                                0_usize,
                            ),
                        )
                        .0;
                    map.alter(&val, |v| (v.0, v.1 + 1));
                    enum_value
                // u32::max_value means something wrong, and 0 means unmapped. And, enum value starts with 1.
                } else {
                    u32::max_value()
                };
                col.push(enum_value);
                col
            })),
            DataType::IpAddr => Column::with_data(records.iter_mut().fold(vec![], |mut col, v| {
                let val: String = v.next().unwrap().iter().map(|&c| c as char).collect();
                col.push(
                    val.parse::<IpAddr>()
                        .unwrap_or_else(|_| IpAddr::V4(Ipv4Addr::new(255, 255, 255, 255))),
                );
                col
            })),
            DataType::DateTime => {
                Column::with_data(records.iter_mut().fold(vec![], |mut col, v| {
                    let val: String = v.next().unwrap().iter().map(|&c| c as char).collect();
                    let fmt = formats.get(&fid).unwrap();
                    col.push(
                        NaiveDateTime::parse_from_str(&val, fmt).unwrap_or_else(|_| {
                            NaiveDateTime::new(
                                NaiveDate::from_ymd(1, 1, 1),
                                NaiveTime::from_hms(0, 0, 0),
                            )
                        }),
                    );
                    col
                }))
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatypes::{DataType, Field};
    use crate::table::convert_to_conc_enum_maps;
    use itertools::izip;

    fn get_test_data() -> (
        Schema,
        Vec<ByteRecord>,
        HashMap<usize, HashMap<String, (u32, usize)>>,
        HashMap<usize, String>,
        Vec<Column>,
    ) {
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

        let mut c5_map: HashMap<u32, String> = HashMap::new();
        c5_map.insert(1, "t1".to_string());
        c5_map.insert(2, "t2".to_string());
        c5_map.insert(7, "t3".to_string());

        let mut records = vec![];
        let fmt = "%Y-%m-%d %H:%M:%S";
        for (c0, c1, c2, c3, c4, c5) in izip!(
            c0_v.iter(),
            c1_v.iter(),
            c2_v.iter(),
            c3_v.iter(),
            c4_v.iter(),
            c5_v.iter()
        ) {
            let mut row: Vec<String> = vec![];
            row.push(c0.to_string());
            row.push(c1.clone());
            row.push(c2.to_string());
            row.push(c3.to_string());
            row.push(c4.format(fmt).to_string());
            row.push(c5_map.get(c5).unwrap().to_string());
            records.push(ByteRecord::from(row));
        }
        let schema = Schema::new(vec![
            Field::new(DataType::Int),
            Field::new(DataType::Str),
            Field::new(DataType::IpAddr),
            Field::new(DataType::Float),
            Field::new(DataType::DateTime),
            Field::new(DataType::Enum),
        ]);
        let mut formats = HashMap::new();
        formats.entry(4).or_insert(fmt.to_string());
        let mut labels = HashMap::new();
        labels.insert(5, c5_map.into_iter().map(|(k, v)| (v, (k, 0))).collect());

        let c0 = Column::from(c0_v);
        let c1 = Column::from(c1_v);
        let c2 = Column::from(c2_v);
        let c3 = Column::from(c3_v);
        let c4 = Column::from(c4_v);
        let c5 = Column::from(c5_v);
        let columns: Vec<Column> = vec![c0, c1, c2, c3, c4, c5];
        (schema, records, labels, formats, columns)
    }

    #[test]
    fn parse_records() {
        let (schema, records, labels, formats, columns) = get_test_data();
        let result = super::records_to_columns(
            records.as_slice(),
            &schema,
            &convert_to_conc_enum_maps(&labels),
            &formats,
        );
        assert_eq!(result, columns);
    }

    #[test]
    fn missing_enum_map() {
        let schema = Schema::new(vec![Field::new(DataType::Enum)]);
        let labels = HashMap::<usize, HashMap<String, (u32, usize)>>::new();

        let row = vec!["1".to_string()];
        let records = vec![ByteRecord::from(row)];

        let result = super::records_to_columns(
            records.as_slice(),
            &schema,
            &convert_to_conc_enum_maps(&labels),
            &HashMap::new(),
        );

        let c = Column::from(vec![u32::max_value()]);
        assert_eq!(c, result[0]);
    }
}
