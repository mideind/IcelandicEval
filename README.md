# IcelandicEval
Utilities to generate Icelandic evaluation data sets for LLMs

## calc-freq.py

This utility program generates evaluation data
for LLMs, typically OpenAI's GPT-4, to test proficiency
in Icelandic. The proficiency tests are mostly about
word inflection and grammatical correctness.

### Usage

The `nouns.csv` and `adjectives.csv` files need to be present
in the `data` directory. They were originally created
by querying the BÍN database
([bin.arnastofnun.is](https://bin.arnastofnun.is)), for example
(in the psql command line):

```bash
psql> \copy (select ord, ofl from bin2023
    where ofl in ('kk', 'kvk', 'hk')) to 'data/nouns.csv' with csv;

psql> \copy (select ord from bin2023 where ofl = 'lo')
    to 'data/adjectives.csv' with csv;
```

Then, given those files, this program is run to generate
randomly sampled, bucketed lists of nouns and adjectives
respectively. The buckets are created by frequency of
occurrence of the word forms in the `icegrams` database,
with bucket 0 containing the least frequent words and bucket
2 the most frequent.

```bash
    python calc-freq.py --nouns
    python calc-freq.py --adjectives
```

Finally, after the buckets 0-2 have been created, the
final evaluation samples can be generated:

```bash
    python calc-freq.py --generate
```

The results are found in the
`data/icelandic-inflection-{easy,medium,hard}.jsonl` files.

# License

Copyright (C) 2023 Miðeind ehf. All rights reserved.

This software is under the MIT License. Consult the LICENSE.md file
for details.

