use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use eyre::{WrapErr};
use sha2::{Sha256, Digest};

pub struct ModelDownloader {
    url: String,
    path: Option<PathBuf>,
    overwrite: bool,
    expected_hash: Option<String>,
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

    pub fn expected_hash(&self) -> &'static str {
        match self {
            KnownModel::Anitag2VecTokenizerV1 => "e155b92198977bb57cd5272265ae66c23be0365d16f92febc568ecce9e89df57",
            KnownModel::Anitag2VecV1 => "5ce2ec0b9873971851702d7161a8f59ab59db919a6dbdd57c7ac9e9dcd04adaf"
        }
    }
}

impl ModelDownloader {
    pub fn new<U: Into<String>>(url: U) -> Self {
        Self {
            url: url.into(),
            path: None,
            expected_hash: None,
            overwrite: false,
        }
    }

    pub fn from_known(known: KnownModel, overwrite: bool) -> Self {
        Self::new(known.url())
            .with_path(known.path())
            .with_expected_hash(known.expected_hash())
            .overwrite(overwrite)
    }

    pub fn with_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.path = Some(path.into());
        self
    }

    pub fn with_expected_hash<S: Into<String>>(mut self, hash: S) -> Self {
        self.expected_hash = Some(hash.into());
        self
    }

    pub fn overwrite(mut self, overwrite: bool) -> Self {
        self.overwrite = overwrite;
        self
    }

    pub fn ensure_checksum_sha256(&self, incoming: &[u8]) -> eyre::Result<()> {
        if let Some(expected_hash) = &self.expected_hash {
            // model is tiny (<=8MB)
            // should be fine
            let mut hasher = Sha256::new();
            hasher.update(incoming);
            let out = hasher.finalize();
            let incoming_hash =hex::encode(out).to_string();
            if expected_hash.ne(&incoming_hash) {
                let filename = self.path.clone().map(|m| m.display().to_string());
                let filename = filename.unwrap_or("<unknown>".to_string());
                eyre::bail!("Expected file {filename} does not match the hash {expected_hash}, got {incoming_hash} instead.");
            }
        }

        Ok(())
    }

    pub fn download(&self) -> eyre::Result<PathBuf> {
        let path = self.resolve_path()?;
        if path.exists() && !self.overwrite {
            let bytes = fs::read(&path).wrap_err("Failed to create file")?;
            self.ensure_checksum_sha256(&bytes)?;
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
        self.ensure_checksum_sha256(&bytes)?;

        let mut file = fs::File::create(&path).wrap_err("Failed to create file")?;
        file.write_all(&bytes).wrap_err("Failed to write file")?;

        Ok(path)
    }

    fn resolve_path(&self) -> eyre::Result<PathBuf> {
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
