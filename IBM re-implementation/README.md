(Partly) Implementation of the system described in "Stance Classification of Context-Dependent Claims" from Bar-Haim et al.

Takes three arguments when run:
1. Path to the claim dataset
2. Mode (0: training the target identifier, 1: testing, 2: testing (& caching))
3. Path to a word2Vec Model (ie. Model trained on Google News: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)