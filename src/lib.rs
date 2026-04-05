pub mod tagtok;
pub mod model;

#[cfg(test)]
mod tests {
    use crate::tagtok::TagSet;
    use super::*;

    #[test]
    fn test_tokenizer() -> eyre::Result<()>{
        let tagtok = tagtok::TagTok::load_from_pytokenizer_v1("src/tests/bpe_example.json")?;
        {
            let out = tagtok.encode_batch([
                TagSet::new(["Hello", "World"]),
                TagSet::new(["Does it work?"]),
            ], None)
            .map_err(|e| eyre::eyre!(e))?;
            let python_tagtok = [
                vec![41, 227, 197, 1, 56, 185, 1764],
                vec![37, 79, 183, 2744, 4476, 32]
            ];

            debug_assert_eq!(out, python_tagtok);
        }
        {
            let imax = 10;
            let out = tagtok.encode_batch([
                TagSet::new(["Hello", "World"]),
                TagSet::new(["Does it work?"]),
            ], Some(imax))
            .map_err(|e| eyre::eyre!(e))?;
            let python_tagtok_padded = [
                vec![41, 227, 197, 1, 56, 185, 1764, 0, 0, 0],
                vec![37, 79, 183, 2744, 4476, 32, 0, 0, 0, 0]
            ];

            assert!(out[0].len() == imax);
            assert!(out[1].len() == imax);
            debug_assert_eq!(out, python_tagtok_padded);
        }

        Ok(())
    }

    #[test]
    fn test_inference() -> eyre::Result<()>{
        let model_path = "checkpoints/anitag2vec_63fc21b89723d1ce_b0d065e705028cb3_i128_e30_s157043_b256_p1871744.onnx";
        let tokenizer_path = "checkpoints/token_dataset_b0d065e705028cb3_vocab_size_5000_freq_3.json";

        let mut anitag2vec = model::Anitag2Vec::load_from_file_v1(model_path, tokenizer_path)?;
        let example = vec![
            TagSet::new(["1girl", "1boy"])
        ];
        anitag2vec.run_inference(example)?;

        eyre::bail!("Hell nah");
        Ok(())
    }
}
