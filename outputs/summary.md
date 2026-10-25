Documents per condition: 20

## Mean CER (lower is better)
| condition   |   gray |   otsu |   enhanced |   enhancedClahe |   enhancedBinary |
|:------------|-------:|-------:|-----------:|----------------:|-----------------:|
| clean       |  0.015 |  0.034 |      0.015 |           0.015 |            0.313 |
| blur        |  0.024 |  0.657 |      0.019 |           0.017 |            0.258 |
| noise       |  0.03  |  0.137 |      0.047 |           0.093 |            0.339 |
| unevenLight |  0.017 |  0.458 |      0.015 |           0.015 |            0.312 |
| lowContrast |  0.017 |  0.03  |      0.071 |           0.068 |            0.274 |
| skew        |  0.264 |  0.266 |      0.016 |           0.016 |            0.039 |
| combined    |  1     |  0.992 |      0.937 |           0.944 |            0.983 |

## Mean WER (lower is better)
| condition   |   gray |   otsu |   enhanced |   enhancedClahe |   enhancedBinary |
|:------------|-------:|-------:|-----------:|----------------:|-----------------:|
| clean       |  0.086 |  0.115 |      0.086 |           0.086 |            0.583 |
| blur        |  0.322 |  0.936 |      0.35  |           0.359 |            0.781 |
| noise       |  0.204 |  0.579 |      0.364 |           0.604 |            0.641 |
| unevenLight |  0.089 |  0.532 |      0.086 |           0.087 |            0.581 |
| lowContrast |  0.092 |  0.116 |      0.269 |           0.263 |            0.671 |
| skew        |  0.357 |  0.377 |      0.092 |           0.095 |            0.202 |
| combined    |  1     |  0.999 |      0.992 |           0.996 |            1     |

## Five worst OCR results across all methods
| condition   | method         |   docIndex |   cer | ocrText   |
|:------------|:---------------|-----------:|------:|:----------|
| combined    | gray           |          0 |     1 |           |
| combined    | otsu           |          0 |     1 |           |
| combined    | enhanced       |          0 |     1 |           |
| combined    | enhancedClahe  |          0 |     1 |           |
| combined    | enhancedBinary |          0 |     1 |           |