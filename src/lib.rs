pub mod tagtok;
pub mod model;
pub mod downloader;

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::{downloader::{ModelDownloader, KnownModel}, tagtok::TagSet};
    use super::*;

    #[test]
    fn test_tokenizer() -> eyre::Result<()>{
        let tokenizer_path = ModelDownloader::from_known(downloader::KnownModel::Anitag2VecTokenizerV1, false).download().unwrap();
        let tagtok = tagtok::TagTok::load_from_pytokenizer_v1(tokenizer_path)?;
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
    fn test_position_generator() -> eyre::Result<()> {
        let tokenizer_path = ModelDownloader::from_known(downloader::KnownModel::Anitag2VecTokenizerV1, false).download().unwrap();
        let tagtok = tagtok::TagTok::load_from_pytokenizer_v1(tokenizer_path)?;
        let arr =  ndarray::Array2::<i64>::from_shape_vec((4, 10), vec![
            1i64, 9, 9, 1, 9, 9, 9, 1, 9, 1,
            9, 9, 9, 9, 9, 1, 9, 9, 9, 9,
            0, 1, 9, 9, 9, 0, 9, 9, 9, 0,
            9, 9, 9, 9, 9, 9, 9, 9, 9, 9
        ])?;

        let pos = tagtok.get_chunked_positions(&arr);
        let expected =  ndarray::Array2::<i64>::from_shape_vec((3, 10), vec![
            0i64, 1, 2, 0, 1, 2, 3, 0, 1, 0,
            1, 2, 3, 4, 5, 0, 1, 2, 3, 4,
            0, 0, 1, 2, 3, 0, 1, 2, 3, 0,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        ])?;

        assert_eq!(pos, expected);

        Ok(())
    }

    #[test]
    fn test_inference_simple() -> eyre::Result<()>{
        let model_path = ModelDownloader::from_known(KnownModel::Anitag2VecV1, false).download().unwrap();
        let tokenizer_path = ModelDownloader::from_known(KnownModel::Anitag2VecTokenizerV1, false).download().unwrap();
        let mut anitag2vec = model::Anitag2Vec::load_from_file_v1(model_path, tokenizer_path)?;
        
        let emb = anitag2vec.run_inference(vec![
            TagSet::new(["Comedy", "TV", "Anime", "Romance"])
        ])?;
        assert_eq!(emb.shape(), [1, 128]);
        assert_eq!(emb.clone().to_vec()[0].len(), 128);

        let emb = &emb.to_vec()[0];
        let head = &emb[..5];
        let tail = &emb[(128-5)..];
        assert_vec_close(head,&[-2.4992497, 2.2522116, 0.9088446, -3.7856572, 0.9309975], 1e-5);
        assert_vec_close(tail, &[1.3903112, -1.0986532, 3.2572346, -2.1192505, -4.5961003], 1e-5);

        Ok(())
    }

    #[test]
    fn test_inference_permutation_invariance() -> eyre::Result<()>{
        let model_path = ModelDownloader::from_known(KnownModel::Anitag2VecV1, false).download().unwrap();
        let tokenizer_path = ModelDownloader::from_known(KnownModel::Anitag2VecTokenizerV1, false).download().unwrap();
        let mut anitag2vec = model::Anitag2Vec::load_from_file_v1(model_path, tokenizer_path)?;
        
        let example = ["cat", "dog", "bird", "unrelated"]
            .into_iter()
            .permutations(4)
            .map(TagSet::new)
            .collect::<Vec<_>>();
        assert_eq!(example.len(), 24);

        let nitems = example.len();
        let emb = anitag2vec.run_inference(example)?;
        assert_eq!(emb.shape(), [nitems, 128]);
        assert_eq!(emb.clone().to_vec()[1].len(), 128);

        // ALL permutations should produce near close embeddings
        let sims = emb.map(|xs| {
            use ndarray::linalg::Dot;
            xs.dot(&xs.t()).into_iter().collect::<Vec<_>>()
        });

        let repr = sims[0];
        for (pos, entry) in sims.iter().enumerate() {
            assert!((entry - repr).abs() < 1e-3, "{entry} != {repr} at {pos}");
        }

        Ok(())
    }

    fn assert_vec_close(a: &[f32], b: &[f32], eps: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x - y).abs() <= eps,
                "mismatch at {}: {} vs {}",
                i, x, y
            );
        }
    }
}
