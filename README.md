# IcelandicEval
A repository of utilities to generate Icelandic
evaluation data sets for LLMs. The data sets are mostly about
word inflection and grammatical correctness.

## calc-freq.py

This utility program generates evaluation data
for LLMs, typically OpenAI's GPT-4, to test proficiency
in Icelandic. The data consists of lists of noun phrases,
where each phrase contains an adjective and a noun,
and the task is to inflect the adjective and noun together
in all four cases (nominative, accusative, dative, genitive),
in singular as well as plural.

The final output of the program is a set of three JSONL
files, each containing a number of samples. The samples are
bucketed into three categories, easy, medium and hard,
depending on the frequency of the adjectives and nouns used
in each sample. Each sample is an LLM prompt and an ideal
completion.

### Usage

Clone this repo into a directory, create a virtualenv and
install the requirements:

```bash
    git clone https://github.com/mideind/IcelandicEval.git
    cd IcelandicEval
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
```

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
2 the most frequent. The bucket files are created in the `data`
directory, under the names `nouns-{0,1,2}.txt` and `adj-{0,1,2}.txt`.

```bash
    python calc-freq.py --nouns
    python calc-freq.py --adjectives
```

Finally, after the buckets 0-2 have been created, the
final evaluation samples can be generated. The number
of samples desired from each bucket can be passed in as
a command line parameter, defaulting to 20.

```bash
    python calc-freq.py --generate [N, default 20]
```

The results are found in three JSONL files, named
`data/icelandic-inflection-{easy,medium,hard}/samples.jsonl`.
They are in a format that is suitable for use with OpenAI's
evals suite (see [github.com/openai/evals](https://github.com/openai/evals)).

# License

Copyright (C) 2023 Miðeind ehf. All rights reserved.

This software is under the MIT License. Consult the LICENSE.md file
for details.

