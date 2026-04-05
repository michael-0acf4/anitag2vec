use std::path::Path;
use ort::{session::{Session, builder::GraphOptimizationLevel}, value::Tensor};
use crate::tagtok::TagTok;

pub struct Anitag2Vec {
    tagtok: TagTok,
    model: Session
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

    pub fn run_inference(&mut self, tag_sets: Vec<Vec<String>>) -> eyre::Result<()> {
        const I_DIM: usize = 128;
        let b_count = tag_sets.len();

        let token_ids = self.tagtok.encode_batch(tag_sets)?
            .into_iter()
            .map(|xs| {
                let mut inp = vec![0_i64; I_DIM];
                let xs = xs.into_iter().map(|x| x as i64).collect::<Vec<_>>();
                let n = I_DIM.min(xs.len());
                inp[..n].copy_from_slice(&xs);
                inp
            })
            .flatten()
            .collect::<Vec<_>>();
        let token_ids = ndarray::Array2::<i64>::from_shape_vec((b_count, I_DIM), token_ids)?;
        println!("{:?}", token_ids.shape());
        let batches = Tensor::from_array(token_ids)?;

        let inputs: Vec<(std::borrow::Cow<'_, str>, ort::session::SessionInputValue<'_>)> = ort::inputs! {
            "x" => batches
        };
        let outputs = self.model.run(inputs)?;
        let Ok(tensor_output): ort::Result<ndarray::ArrayViewD<f32>> = outputs[0].try_extract_array() else {
            eyre::bail!("First output was not a Tensor<f32>!");
        };
        let shape = tensor_output.shape();
        println!("Output shape {shape:?}: {tensor_output}");

        Ok(())
    }
}
