use std::{path::Path};
use ahash::AHashMap;
use eyre::Context;
use serde::Deserialize;
use tokenizers::{AddedToken, Tokenizer, models::bpe::BPE};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;

pub struct TagTok {
    bpe: Tokenizer
}

#[derive(Deserialize)]
struct DescrAdded {
    #[allow(unused)]
    id: u32,
    content: String,
    special: bool,
}

#[derive(Deserialize)]
struct DescrDecoder {
    #[serde(rename = "type")]
    _type: String,
    add_prefix_space: bool,
    trim_offsets: bool,
    use_regex: bool
}

#[derive(Deserialize)]
struct DescrModel {
    vocab: AHashMap<String, u32>,
    merges: Vec<(String, String)>,
}

#[derive(Deserialize)]
struct DescrContent {
    added_tokens: Vec<DescrAdded>,
    pre_tokenizer: DescrDecoder,
    decoder: DescrDecoder,
    model: DescrModel,
}

impl TagTok {
    pub fn load_from_pytokenizer_v1<P: AsRef<Path>>(path: P) -> eyre::Result<Self>{
        let content = std::fs::read_to_string(path)?;
        let trained = serde_json::from_str::<DescrContent>(&content)
            .wrap_err_with(|| eyre::eyre!("Could not parse trained BPE model"))?;
        if trained.decoder._type.ne("ByteLevel") || trained.pre_tokenizer._type.ne("ByteLevel") {
            eyre::bail!("Expected decoder to be of type ByteLevel")
        }

        let pre_tokenizer = ByteLevel::default()
            .add_prefix_space(trained.pre_tokenizer.add_prefix_space)
            .trim_offsets(trained.pre_tokenizer.trim_offsets)
            .use_regex(trained.pre_tokenizer.use_regex); // !
        let decoder =  ByteLevel::default()
            .add_prefix_space(trained.decoder.add_prefix_space)
            .trim_offsets(trained.decoder.trim_offsets)
            .use_regex(trained.decoder.use_regex); // !

        let added_tokens = {
            trained
                .added_tokens
                .into_iter()
                .map(|tk| {
                    AddedToken::from(tk.content, tk.special)
                })
                .collect::<Vec<_>>()
        };

        let mut bpe = Tokenizer::new( BPE::new(trained.model.vocab, trained.model.merges));
        bpe.add_special_tokens(&added_tokens);
        bpe.with_pre_tokenizer(Some(pre_tokenizer));
        bpe.with_decoder(Some(decoder));

        Ok(Self { bpe })
    }

    pub fn encode_batch(&self, sequences: Vec<Vec<String>>) -> eyre::Result<Vec<Vec<u32>>> {
        let encodings = self.bpe.encode_batch(
            sequences,
            true, // add_special_tokens
        ).map_err(|e| eyre::eyre!(e))?;

        let ret = encodings
            .into_iter()
            .map(|encoding| encoding.get_ids().to_vec())
            .collect::<Vec<Vec<_>>>();

        Ok(ret)
    }
}
