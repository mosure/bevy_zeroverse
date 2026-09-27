use serde::Serialize;

pub(crate) const RESPONSE_SCHEMA_VERSION: u32 = 1;
pub(crate) const IMAGE_EMBEDDING_METHOD: &str = "encodeImageBytesJson";
pub(crate) const TEXT_EMBEDDING_METHOD: &str = "encodeTextsJson";
pub(crate) const SCORE_METHOD: &str = "scoreImageTextsJson";

#[derive(Serialize)]
pub(crate) struct EmbeddingResponse {
    pub(crate) schema_version: u32,
    pub(crate) method: &'static str,
    pub(crate) shape: [usize; 2],
    pub(crate) raw_embedding: Vec<f32>,
    pub(crate) normalized_embedding: Vec<f32>,
}

#[derive(Serialize)]
pub(crate) struct ScoreResponse {
    pub(crate) schema_version: u32,
    pub(crate) method: &'static str,
    pub(crate) image_embedding_shape: [usize; 2],
    pub(crate) text_embedding_shape: [usize; 2],
    pub(crate) logits_shape: [usize; 2],
    pub(crate) raw_image_embedding: Vec<f32>,
    pub(crate) normalized_image_embedding: Vec<f32>,
    pub(crate) raw_text_embedding: Vec<f32>,
    pub(crate) normalized_text_embedding: Vec<f32>,
    pub(crate) logits_per_image: Vec<f32>,
    pub(crate) probabilities_per_image: Vec<f32>,
}

pub(crate) fn serialize_response(response: &impl Serialize) -> Result<String, serde_json::Error> {
    serde_json::to_string(response)
}

#[cfg(test)]
mod tests {
    use super::{
        EmbeddingResponse, IMAGE_EMBEDDING_METHOD, RESPONSE_SCHEMA_VERSION, SCORE_METHOD,
        ScoreResponse, TEXT_EMBEDDING_METHOD, serialize_response,
    };

    #[test]
    fn embedding_response_json_contract_is_stable() -> Result<(), serde_json::Error> {
        let json = serialize_response(&EmbeddingResponse {
            schema_version: RESPONSE_SCHEMA_VERSION,
            method: IMAGE_EMBEDDING_METHOD,
            shape: [1, 2],
            raw_embedding: vec![3.0, 4.0],
            normalized_embedding: vec![0.6, 0.8],
        })?;

        assert_eq!(
            json,
            r#"{"schema_version":1,"method":"encodeImageBytesJson","shape":[1,2],"raw_embedding":[3.0,4.0],"normalized_embedding":[0.6,0.8]}"#
        );
        Ok(())
    }

    #[test]
    fn text_embedding_response_uses_the_exported_method_name() -> Result<(), serde_json::Error> {
        let json = serialize_response(&EmbeddingResponse {
            schema_version: RESPONSE_SCHEMA_VERSION,
            method: TEXT_EMBEDDING_METHOD,
            shape: [2, 1],
            raw_embedding: vec![1.0, 2.0],
            normalized_embedding: vec![1.0, 1.0],
        })?;

        assert_eq!(
            json,
            r#"{"schema_version":1,"method":"encodeTextsJson","shape":[2,1],"raw_embedding":[1.0,2.0],"normalized_embedding":[1.0,1.0]}"#
        );
        Ok(())
    }

    #[test]
    fn score_response_preserves_logits_and_adds_all_embeddings() -> Result<(), serde_json::Error> {
        let json = serialize_response(&ScoreResponse {
            schema_version: RESPONSE_SCHEMA_VERSION,
            method: SCORE_METHOD,
            image_embedding_shape: [1, 2],
            text_embedding_shape: [2, 2],
            logits_shape: [1, 2],
            raw_image_embedding: vec![1.0, 2.0],
            normalized_image_embedding: vec![0.25, 0.5],
            raw_text_embedding: vec![3.0, 4.0, 5.0, 6.0],
            normalized_text_embedding: vec![0.3, 0.4, 0.5, 0.6],
            logits_per_image: vec![7.0, 8.0],
            probabilities_per_image: vec![0.9, 0.8],
        })?;

        assert_eq!(
            json,
            r#"{"schema_version":1,"method":"scoreImageTextsJson","image_embedding_shape":[1,2],"text_embedding_shape":[2,2],"logits_shape":[1,2],"raw_image_embedding":[1.0,2.0],"normalized_image_embedding":[0.25,0.5],"raw_text_embedding":[3.0,4.0,5.0,6.0],"normalized_text_embedding":[0.3,0.4,0.5,0.6],"logits_per_image":[7.0,8.0],"probabilities_per_image":[0.9,0.8]}"#
        );
        Ok(())
    }
}
