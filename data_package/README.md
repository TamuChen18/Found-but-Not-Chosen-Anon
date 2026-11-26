# Data package

Raw candidate pools, per-request logs and candidate-level evaluation results for the paper.
The package is split into two zip archives because of the GitHub upload size limit. Each archive is
self-contained and can be opened on its own, but you need both for full reproduction.

| File | Size |
|---|---|
| `found-but-not-chosen-data-anon-part1-of-2.zip` | 22.6 MB |
| `found-but-not-chosen-data-anon-part2-of-2.zip` | 19.5 MB |
| `SHA256SUMS.txt` | SHA-256 checksums of the two archives |

## Usage

From the repository root:

```bash
# optional: verify the downloads
(cd data_package && sha256sum -c SHA256SUMS.txt)

# extract both archives into the repository root
unzip data_package/found-but-not-chosen-data-anon-part1-of-2.zip -d .
unzip data_package/found-but-not-chosen-data-anon-part2-of-2.zip -d .

# reproduce all paper numbers, tables and figures (offline, no GPU)
bash scripts/reproduce_all.sh
```

After extraction, the repository root contains:

```
runs/        generation configs and raw candidate pools of every run
logs/        per-request call logs (sampling parameters, finish reasons, token usage;
             no prompts, completions or credentials)
artifacts/   raw prompt-matched pools and candidate-level result files
licenses/    license of the DisasterBench_Open task data
DATA_NOTICE.md
```

## Note

This data was omitted from the original supplementary material and was added during the
author-response period. The two archives repackage, without modification, the files of the data
package prepared before submission.
