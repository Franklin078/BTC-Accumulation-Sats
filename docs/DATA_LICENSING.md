# Data licensing

The code in this repository is MIT licensed; the data files are not, and this note records
what governs each of them and why committing them is permitted.

## CoinMetrics community series (`data/Coin Metrics/coinmetrics_btc.csv`)

Coin Metrics community data is provided under the Creative Commons
Attribution-NonCommercial 4.0 International licence (CC BY-NC 4.0), which permits copying and
redistribution with attribution, a licence link, and an indication of changes, for
non-commercial purposes (terms: https://coinmetrics.io/terms-of-use/; licence:
https://creativecommons.org/licenses/by-nc/4.0/). This repository redistributes the community
daily series for Bitcoin unmodified apart from column selection, with attribution here and in
the README, for non-commercial academic research. Anyone reusing this file inherits the same
terms; commercial use requires Coin Metrics' own licensing.

## Frozen 2025 tournament dataset (`data/official_2025/stacking_sats_data.parquet`)

This file is vendored byte-identical (SHA-256 recorded in `docs/UPSTREAM_HASHES.txt`) from the
Trilemma Foundation's public, MIT-licensed 2025 tournament repository, which distributes it as
the challenge's official dataset. It is kept solely so the published tournament reference
numbers can be reproduced exactly. Its contents aggregate several providers (CoinMetrics
on-chain series, exchange market data, macro series, a sentiment index), and each provider's
own terms continue to apply to its columns; this repository adds no rights beyond mirroring
what the challenge itself distributes, and only the CoinMetrics price column is ever used for
scoring.

## Fear and Greed index (`data/fear_greed.csv`)

Cached from the alternative.me open API, which offers the index without charge and asks for
attribution. Used in one registered modelling round (a negative result); retained so that
round reproduces.
