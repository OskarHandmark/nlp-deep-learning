# Deep learning for relation extraction example

See https://backtick.se/blog/whatevs

## Installation
Requires Docker, Tested with Python 3.6.2. Note that Tensorflow does not support Python 3.7 yet.

```bash
pip install -r req.txt
```

## Running
```bash
# start StanfordCoreNLP service on port 9000
docker-compose up -d

# Train model
python train.py

# Give it a sentence
python predict.py "I was impressed by the battery life"

# outputs:

      -  b.DESC  b.ENT  i.DESC  i.ENT       word    vote
---------------------------------------------------------      
0  0.51    0.12   0.28    0.07   0.03          I       -
1  0.70    0.03   0.14    0.07   0.06        was       -
2  0.03    0.94   0.01    0.02   0.00  impressed  b.DESC
3  0.85    0.04   0.04    0.04   0.03         by       -
4  0.78    0.02   0.02    0.12   0.06        the       -
5  0.06    0.13   0.61    0.17   0.03    battery   b.ENT
6  0.24    0.01   0.06    0.09   0.60       life   i.ENT
```