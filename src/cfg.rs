// Zama whitepaper 80 bit security key parameters
/*
pub const LWE_DIMENSION: usize = 592;
pub const LWE_NOISE: i32 = -23;
pub const RLWE_SIZE: usize = 2048;
pub const RLWE_NOISE: i32 = -60;
pub const BASE_LOG: usize = 6;
pub const LEVEL: usize = 4;
*/

pub const PRECISION: usize = 6; // quantization bits of inputs and bootstrapped values
pub const WEIGHT_PRECISION: usize = 6; // quantization bits of weights
pub const BOOTSTRAP_INTERVAL_SCALING: f64 = 1.1; // additional scaling on encryption intervals to prevent overflow
pub const SCALE_HINT_SCALING: f64 = 1.1 * 1.1 * 2.; // additional scaling on heuristic scale hint to prevent overflow

// Zama whitepaper 128 bit security key parameters

pub const LWE_DIMENSION: usize = 938;
pub const LWE_NOISE: i32 = -23;
pub const RLWE_SIZE: usize = 4096;
pub const RLWE_NOISE: i32 = -62;
pub const BASE_LOG: usize = 6;
pub const LEVEL: usize = 4;
