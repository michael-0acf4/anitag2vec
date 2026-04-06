use anitag2vec::{downloader::{ModelDownloader, KnownModel}, model::Anitag2Vec, tagtok::TagSet};

fn main() {
    println!("Downloading models...");
    let model_path = ModelDownloader::from_known(KnownModel::Anitag2VecV1, false).download().unwrap();
    let tokenizer_path = ModelDownloader::from_known(KnownModel::Anitag2VecTokenizerV1, false).download().unwrap();
    println!("Done!");

    let mut anitag2vec = Anitag2Vec::load_from_file_v1(model_path, tokenizer_path).unwrap();
    let example = vec![
        TagSet::new(["transcend", "uma musume", "imageset", "japanese"]),
        TagSet::new(["Comedy", "TV", "Anime", "Romance"]),
    ];
    let emb = anitag2vec.run_inference(example).unwrap();
    println!("{:?}", emb.shape()); // [2, 128]

    // Similar to emb.map(|nd| ..)
    // This representation allows various math operations
    println!("{}", emb.ndarray());

    // or alternatively as Vec<Vec<f32>>
    println!("{:?}", emb.to_vec());
}
