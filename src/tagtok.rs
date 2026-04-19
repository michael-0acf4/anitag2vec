use std::{path::Path};
use ahash::AHashMap;
use eyre::{Context, ContextCompat};
use serde::Deserialize;
use tokenizers::{AddedToken, Tokenizer, models::bpe::BPE};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;


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

#[derive(Clone, Debug)]
pub struct TagSet {
    set: Vec<String>
}

impl TagSet {
    pub fn new<I, S>(items: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String> {
        Self {
            set: items.into_iter().map(Into::into).collect(),
        }
    }
}

pub struct TagTok {
    bpe: Tokenizer,
    pad_token_id: u32,
    sep_token_id: u32,
}

impl TagTok {
    pub fn load_from_pytokenizer_v1<P: AsRef<Path>>(path: P) -> eyre::Result<Self>{
        let content = std::fs::read_to_string(path)?;
        let trained = serde_json::from_str::<DescrContent>(&content)
            .wrap_err("Could not parse trained BPE model")?;
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

        let pad = bpe.id_to_token(0).wrap_err("Invalid tokenizer state")?;
        let sep = bpe.id_to_token(1).wrap_err("Invalid tokenizer state")?;
        if pad.ne("[PAD]") {
            eyre::bail!("Expected [PAD] token to be at index 0")
        }
        if sep.ne("[SEP]") {
            eyre::bail!("Expected [SEP] token to be at index 1")
        }

        Ok(Self { bpe, pad_token_id: 0, sep_token_id: 1 })
    }

    pub fn encode(&self, tags: TagSet, pad_fixed_size: Option<usize>) -> eyre::Result<Vec<u32>> {
        let mut out = vec![];
        let count = tags.set.len();
        for (i, set) in tags.set.into_iter().enumerate() {
            let output = self.bpe
                .encode(set, false)
                .map_err(|e| eyre::eyre!(e))?
                .get_ids()
                .to_vec();
            out.extend(output);
            if i < count - 1 {
                out.push(self.sep_token_id);
            }
        }

        if let Some(imax) = pad_fixed_size {
            if out.len() > imax {
                out.truncate(imax);
                return Ok(out);
            }

            let left = imax - out.len();
            let pad_id = self.pad_token_id;
            let padding = vec![pad_id; left];
            out.extend(padding);
        }

        Ok(out)
    }

    pub fn encode_batch<I>(&self, tags: I, pad_fixed_size: Option<usize>) -> eyre::Result<Vec<Vec<u32>>>
    where 
        I: IntoIterator<Item = TagSet>
     {
        tags
            .into_iter()
            .map(|tags| self.encode(tags, pad_fixed_size))
            .collect::<eyre::Result<Vec<_>>>()
    }

    pub fn get_chunked_positions(&self, tokens: &ndarray::Array2<i64>) -> ndarray::Array2<i64> {
        let (rows, cols) = tokens.dim();
        let mut pos = ndarray::Array2::<i64>::zeros((rows, cols));
        let spe_tokens = [self.sep_token_id as i64, self.pad_token_id as i64];
        for (mut pos_row, token_row) in pos
            .axis_iter_mut(ndarray::Axis(0))
            .zip(tokens.axis_iter(ndarray::Axis(0)))
        {
            let mut current = 1i64;
            for t in 0..cols {
                if spe_tokens.contains(&token_row[t])  {
                    pos_row[t] = 0;
                    current = 1;
                } else {
                    pos_row[t] = current;
                    current += 1;
                }
            }
        }
        pos
    }
}
