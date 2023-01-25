# MNIST demo
This is a step-by-step manual to demonstrate the interface of `he-man-concrete` by classifying MNIST images.<br/>

**Step 0: Build he-man-concrete**<br/>
```
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

**Step 1: Keyparams Generation**<br/>
The model owner calls `keyparams` with the following parameters
- model
- calibration data
- keyparams output file
- calibrated output model
```
target/release/he-man-concrete keyparams demo/mnist.onnx demo/calibration-data.zip demo/keyparams.json demo/mnist_calibrated.onnx
```

**Step 2: Key Generation**<br/>
The client calls `keygen` with the following parameters
- keyparameter file
- output directory for keys
```
target/release/he-man-concrete keygen demo/keyparams.json demo/
```

**Step 3: Encryption**<br/>
Next, the client calls `encrypt` with
- key directory
- input to be encrypted
- encrypted ciphertext output
```
target/release/he-man-concrete encrypt demo/ demo/input.npy demo/input.enc
```

**Step 4: Inference**<br/>
The model owner performs `inference` with
- key directory
- calibrated model
- input ciphertext
- encrypted output ciphertext
```
/target/release/he-man-concrete inference demo/ demo/mnist_calibrated.onnx demo/input.enc demo/result.enc
```

**Step 5: Decryption**<br/>
Lastly, the client decrypts the encrypted result using `decrypt` with
- key directory
- ciphertext to be decrypted
- decrypted output result
```
target/release/he-man-concrete decrypt demo/ demo/result.enc demo/result.npy
```