use crate::{cfg, io};

use concrete::*;
use serde::{Deserialize, Serialize};
// use std::time::Instant;

pub struct KeyChain {
    pub secret_key: SecretKey,
    pub evaluation_key: EvaluationKey,
}

impl KeyChain {
    pub fn _load(path: &str) -> Self {
        let secret_key = SecretKey::load(path);
        let evaluation_key = EvaluationKey::load(path);

        KeyChain {
            secret_key,
            evaluation_key,
        }
    }

    pub fn save(self, path: &str) {
        self.secret_key.save(path);
        self.evaluation_key.save(path);
    }

    pub fn generate(params: KeyParams) -> Self {
        // let start_time = Instant::now();

        // println!("Generating keys...");
        let rlwe_params = RLWEParams {
            dimension: 1,
            polynomial_size: cfg::RLWE_SIZE,
            log2_std_dev: cfg::RLWE_NOISE,
        };
        let lwe_params = LWEParams {
            dimension: cfg::LWE_DIMENSION,
            log2_std_dev: cfg::LWE_NOISE,
        };
        let sk_rlwe = RLWESecretKey::new(&rlwe_params);
        let sk_in = LWESecretKey::new(&lwe_params);
        let sk_out = sk_rlwe.to_lwe_secret_key();
        // create bootstrapping key
        let bsk = LWEBSK::new(&sk_in, &sk_rlwe, cfg::BASE_LOG, cfg::LEVEL);
        // create key switching key
        let ksk = LWEKSK::new(&sk_out, &sk_in, cfg::BASE_LOG, cfg::LEVEL);

        // println!(
        //     "Generating keys took {} seconds",
        //     start_time.elapsed().as_secs_f64()
        // );

        KeyChain {
            secret_key: SecretKey {
                secret_key_in: sk_in,
                secret_key_out: sk_out,
                params,
            },
            evaluation_key: EvaluationKey {
                bootstrapping_key: bsk,
                key_switching_key: ksk,
            },
        }
    }
}

#[derive(Deserialize, Serialize)]
pub struct KeyParams {
    pub input_min: f64,
    pub input_max: f64,
}

#[derive(Serialize, Deserialize)]
pub struct SecretKey {
    pub secret_key_in: LWESecretKey,
    pub secret_key_out: LWESecretKey,
    pub params: KeyParams
}

impl SecretKey {
    pub fn load(path: &str) -> SecretKey {
        // let start_time = Instant::now();
        // println!("Loading secret key...");

        let path = std::path::PathBuf::from(path);
        let secret_key = io::load_serialized(path.join("secret_key.bin").to_str().unwrap());

        // println!(
        //     "Loading secret key took {} seconds",
        //     start_time.elapsed().as_secs_f64()
        // );

        secret_key
    }

    pub fn save(self, path: &str) {
        let path = std::path::PathBuf::from(path);
        io::save_serialized(path.join("secret_key.bin").to_str().unwrap(), self);
    }
}

#[derive(Serialize, Deserialize)]
pub struct EvaluationKey {
    pub bootstrapping_key: LWEBSK,
    pub key_switching_key: LWEKSK,
}

impl EvaluationKey {
    pub fn load(path: &str) -> EvaluationKey {
        // let start_time = Instant::now();
        // println!("Loading evaluation key...");

        let path = std::path::PathBuf::from(path);
        let evaluation_key = io::load_serialized(path.join("public_key.bin").to_str().unwrap());

        // println!(
        //     "Loading evaluation key took {} seconds",
        //     start_time.elapsed().as_secs_f64()
        // );

        evaluation_key
    }

    pub fn save(self, path: &str) {
        let path = std::path::PathBuf::from(path);
        io::save_serialized(path.join("public_key.bin").to_str().unwrap(), self);
    }
}
