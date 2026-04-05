use std::path::Path;
use crate::tagtok::{TagSet, TagTok};
use tract_onnx::{prelude::*};
use tract_ndarray::Array2;

type Model = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;
pub struct Anitag2Vec {
    tagtok: TagTok,
    model: Model
}

impl Anitag2Vec {
    pub fn load_from_file_v1<P: AsRef<Path>>(
        onnx_model: P,
        tokenizer: P,
    ) -> eyre::Result<Self> {
        let tagtok = TagTok::load_from_pytokenizer_v1(tokenizer)?;
        let model = tract_onnx::onnx()
            .model_for_path(onnx_model)
            .map_err(|e| eyre::eyre!(e))?
            .into_optimized()
            .map_err(|e| eyre::eyre!(e))?
            .into_runnable()
            .map_err(|e| eyre::eyre!(e))?;

        Ok(Self { tagtok, model })
    }

    pub fn run_inference(&mut self, tag_sets: Vec<TagSet>) -> eyre::Result<()> {
        const I_DIM: usize = 128;
        let b_count = tag_sets.len();
        let token_ids = self.tagtok
            .encode_batch(tag_sets, Some(I_DIM))?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        let token_ids = Array2::from_shape_vec((b_count, I_DIM), token_ids)?;
        println!("{:?}", token_ids.shape());
        // let runn = self.model.run(token_ids);

        Ok(())
    }
}
