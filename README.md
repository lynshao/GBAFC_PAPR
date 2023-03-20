# GBAFC_PAPR: Evaluating the PAPR of GBAFC

## Training
```train
python main.py --snr1 [input] --snr2 [input] --train 1 --batchSize [input]
```

## Test
```train
python main.py --snr1 [input] --snr2 [input] --precoding [input] --mapping [input] --train 0 --batchSize [input] --clip [input]
```
Available well-trained models:
| Feedforwad SNR | Feedback SNR |
| ------------- | ------------- |
| -1.5 dB  | 100 dB  |
| -1 dB  | 100 dB  |
| 0 dB  | 20 dB  |

Configurations:
|   | precoding | mapping |
| ------------- | ------------- |  ------------- |
| OFDMA  | 0  | 0  |
| LFDMA  | 1  | 1  |

## Results
![Alt text](PAPR.jpg?raw=true "Title")
