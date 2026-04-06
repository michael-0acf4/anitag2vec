use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use eyre::{Result, WrapErr};

pub struct ModelDownloader {
    url: String,
    path: Option<PathBuf>,
    overwrite: bool
}

pub enum KnownModel {
    Anitag2VecTokenizerV1,
    Anitag2VecV1,
}

impl KnownModel {
    pub fn url(&self) -> &'static str {
        match self {
            KnownModel::Anitag2VecTokenizerV1 => "https://huggingface.co/michael-0acf4/anitag2vec/resolve/main/onnx/token_dataset_b0d065e705028cb3_vocab_size_5000_freq_3.json",
            KnownModel::Anitag2VecV1 => "https://huggingface.co/michael-0acf4/anitag2vec/resolve/main/onnx/anitag2vec_63fc21b89723d1ce_b0d065e705028cb3_i128_e30_s157043_b256_p1871744.onnx"
        }
    }

    pub fn path(&self) -> PathBuf {
        match self {
            KnownModel::Anitag2VecTokenizerV1 => PathBuf::from("anitag2vec_tokenizer_v1.json"),
            KnownModel::Anitag2VecV1 => PathBuf::from("anitag2vec_v1.onnx")
        }
    }
}

impl ModelDownloader {
    pub fn new<U: Into<String>>(url: U) -> Self {
        Self {
            url: url.into(),
            path: None,
            overwrite: false,
        }
    }

    pub fn from_known(known: KnownModel, overwrite: bool) -> Self {
        Self::new(known.url())
            .with_path(known.path())
            .overwrite(overwrite)
    }

    pub fn with_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.path = Some(path.into());
        self
    }

    pub fn overwrite(mut self, overwrite: bool) -> Self {
        self.overwrite = overwrite;
        self
    }

    pub fn download(&self) -> Result<PathBuf> {
        let path = self.resolve_path()?;
        if path.exists() && !self.overwrite {
            return Ok(path);
        }

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .wrap_err("Failed to create parent directory")?;
        }

        let resp = reqwest::blocking::get(&self.url).wrap_err("Failed to send request")?;
        if !resp.status().is_success() {
            eyre::bail!("Download failed with status {}", resp.status());
        }

        let bytes = resp.bytes().wrap_err("Failed to read response body")?;
        let mut file = fs::File::create(&path).wrap_err("Failed to create file")?;
        file.write_all(&bytes).wrap_err("Failed to write file")?;

        Ok(path)
    }

    fn resolve_path(&self) -> Result<PathBuf> {
        if let Some(path) = &self.path {
            return Ok(path.clone());
        }

        let filename = self.url
            .split('/')
            .last()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| eyre::eyre!("Could not infer filename from url"))?;

        Ok(Path::new("models").join(filename))
    }
}
