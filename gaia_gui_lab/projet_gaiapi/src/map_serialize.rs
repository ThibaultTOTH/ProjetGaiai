use serde::{Serialize, Serializer};
use crate::map::Map;

impl Serialize for Map {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("Map", 3)?;
        state.serialize_field("count", &self.count)?;
        let coords_slice = &self.coords[..self.count];
        state.serialize_field("coords", coords_slice)?;
        let hexes_slice = &self.hexes[..self.count];
        state.serialize_field("hexes", hexes_slice)?;
        // skip neighbors to save space or serialize if needed
        state.end()
    }
}
