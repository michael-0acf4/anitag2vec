use std::path::Path;
use crate::tagtok::{TagSet, TagTok};
use ndarray::{ArrayBase, Dim, IxDynImpl, OwnedRepr};
use itertools::Itertools;
use tract_onnx::{prelude::*, tract_core::plan::SimplePlan};

type Plan = Arc<SimplePlan<TypedFact, Box<dyn TypedOp>>>;
pub struct Anitag2Vec {
    tagtok: TagTok,
    plan: Plan
}

type OwnedArrayBaseF32 = ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>, f32>;
#[derive(Clone, Debug)]
pub struct Embedding {
    row_dim: usize,
    inner: OwnedArrayBaseF32
}

impl Embedding {
    pub fn map<O, F>(self, f: F) -> O
    where
        F: Fn(OwnedArrayBaseF32) -> O
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

    pub fn ndarray(self) -> OwnedArrayBaseF32 {
        self.inner
    }
}

impl Anitag2Vec {
    pub fn load_from_file_v1<P: AsRef<Path>>(
        onnx_model: P,
        tokenizer: P,
    ) -> eyre::Result<Self> {
        let tagtok = TagTok::load_from_pytokenizer_v1(tokenizer)?;
        let plan = tract_onnx::onnx()
            .model_for_path(onnx_model)
            .map_err(|e| eyre::eyre!(e))?
            .into_optimized()
            .map_err(|e| eyre::eyre!(e))?
            .into_runnable()
            .map_err(|e| eyre::eyre!(e))?;

        Ok(Self { tagtok, plan })
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
        let input_tensor = token_ids.into_tensor();

        let mut outputs = self.plan
            .run(tvec![input_tensor.into()])
            .map_err(|e| eyre::eyre!(e))?;
        let result = outputs.remove(0);
        let tensor: &Tensor = &result;
        let array_view = tensor
            .to_plain_array_view::<f32>()
            .map_err(|e| eyre::eyre!(e))?;

        Ok(Embedding { row_dim: I_DIM, inner: array_view.to_owned() })
    }
}
