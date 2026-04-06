use std::path::Path;
use crate::tagtok::{TagSet, TagTok};
use ndarray::{ArrayBase, Dim, IxDynImpl, OwnedRepr};
use ort::{session::{Session, builder::GraphOptimizationLevel}, value::Tensor};
use itertools::Itertools;

pub struct Anitag2Vec {
    tagtok: TagTok,
    model: Session
}

type OwnedReprF32 = ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>, f32>;
#[derive(Clone, Debug)]
pub struct Embedding {
    row_dim: usize,
    inner: OwnedReprF32
}

impl Embedding {
    pub fn map<O, F>(self, f: F) -> O
    where
        F: Fn(OwnedReprF32) -> O
    {
        f(self.inner)
    }

    pub fn to_vec(self) -> Vec<Vec<f32>> {
        let idim = self.row_dim;
        self.map(|xs| {
            xs
                .into_iter()
                .chunks(idim) // itertool only
                .into_iter()
                .map(|c| c.collect())
                .collect::<Vec<Vec<_>>>()
        })
    }

    pub fn shape(&self) -> &[usize] {
        self.inner.shape()
    }
}

impl Anitag2Vec {
    pub fn load_from_file_v1<P: AsRef<Path>>(
        onnx_model: P,
        tokenizer: P,
    ) -> eyre::Result<Self> {
        let tagtok = TagTok::load_from_pytokenizer_v1(tokenizer)?;
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| eyre::eyre!(e.to_string()))?
            .with_intra_threads(4)
            .map_err(|e| eyre::eyre!(e.to_string()))?
            .commit_from_file(onnx_model)?;

        Ok(Self { tagtok, model })
    }

    pub fn run_inference(&mut self, tag_sets: Vec<TagSet>) -> eyre::Result<Embedding> {
        const I_DIM: usize = 128;
        let b_count = tag_sets.len();
        let token_ids = self.tagtok
            .encode_batch(tag_sets, Some(I_DIM))?
            .into_iter()
            .flatten()
            .into_iter()
            .map(|x| x as i64)
            .collect::<Vec<_>>();

        let token_ids = ndarray::Array2::<i64>::from_shape_vec((b_count, I_DIM), token_ids)?;
        let batches = Tensor::from_array(token_ids)?;
        let inputs: Vec<(std::borrow::Cow<'_, str>, ort::session::SessionInputValue<'_>)> = ort::inputs! {
            "x" => batches
        };
        let outputs = self.model.run(inputs)?;
        let Ok(tensor_output): ort::Result<ndarray::ArrayViewD<f32>> = outputs[0].try_extract_array() else {
            eyre::bail!("First output was not a Tensor<f32>!");
        };

        Ok(Embedding {
            row_dim: I_DIM,
            inner: tensor_output.to_owned()
        })
    }
}
