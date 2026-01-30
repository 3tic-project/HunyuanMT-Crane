#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Size {
    pub longest_edge: usize,
    pub shortest_edge: usize,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct PreprocessorConfig {
    pub size: Size,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub image_mean: Vec<f32>,
    pub image_std: Vec<f32>,
}
