pub mod tagtok;
pub mod model;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer() -> eyre::Result<()>{
        let tagtok = tagtok::TagTok::load_from_pytokenizer_v1("src/tests/bpe_example.json")?;
        let out = tagtok.encode_batch(vec![
            vec!["Hello World".to_string()],
            vec!["Does it work?".to_string()]
        ])?;

        let python_tagtok = [
            [41, 227, 197, 1699, 185, 1764],
            [37, 79, 183, 2744, 4476, 32]
        ];

        debug_assert_eq!(out, python_tagtok);

        Ok(())
    }

    #[test]
    fn test_inference() -> eyre::Result<()>{
        let model_path = "checkpoints/anitag2vec_63fc21b89723d1ce_b0d065e705028cb3_i128_e3_s157043_b256_p1871744.pth";
        let tokenizer_path = "checkpoints/token_dataset_b0d065e705028cb3_vocab_size_5000_freq_3.json";

        let mut anitag2vec = model::Anitag2Vec::load_from_file_v1(model_path, tokenizer_path)?;
        let example = vec![
            vec!["1girl".to_owned(), "1boy".to_owned()]
        ];
        anitag2vec.run_inference(example)?;

        eyre::bail!("Hell nah");
        Ok(())
    }
}
