"""

    Icelandic LLM evaluation data generator

    Copyright (C) 2023 Miðeind ehf.
    All rights reserved.

    This utility program generates evaluation data
    for LLMs, typically OpenAI's GPT-4, to test proficiency
    in Icelandic. The proficiency tests are mostly about
    word inflection and grammatical correctness.

    Usage
    -----

    The nouns.csv and adjectives.csv files need to be present
    in the data directory. They were originally created
    by querying the BÍN database (bin.arnastofnun.is), for
    example (in the psql command line):

    ```bash
    psql> \\copy (select ord, ofl from bin2023
        where ofl in ('kk', 'kvk', 'hk')) to 'data/nouns.csv' with csv;

    psql> \\copy (select ord from bin2023 where ofl = 'lo')
        to 'data/adjectives.csv' with csv;
    ```

    Then, given those files, this program is run to generate
    randomly sampled, bucketed lists of nouns and adjectives
    respectively. The buckets are created by frequency of
    occurrence of the word forms in the icegrams database,
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
    icelandic-inflection-{easy,medium,hard}.jsonl files.

"""

import functools
import json
from typing import IO, Dict, Iterator, List, Mapping, Set, Tuple, FrozenSet

import random
import math

from collections import defaultdict
from pathlib import Path

import icegrams
import islenska
from reynir import NounPhrase

NounTuple = Tuple[str, str]

# Pesky lemmas that seem to get through frequency filtering by
# being word forms of other categories or lemmas
AVOID_ADJECTIVES: FrozenSet[str] = frozenset(["gar"])
AVOID_NOUNS: FrozenSet[str] = frozenset(["nam", "góða", "ex", "virt", "óþörf", "rift"])

MAX_BUCKETS = 3

NUM_NOUN_SAMPLES = 1000
NUM_ADJECTIVE_SAMPLES = 500

DIFFICULTY: Mapping[int, str] = {
    0: "hard",
    1: "medium",
    2: "easy"
}

# Instantiate the icegrams database of unigram, bigram and trigram frequencies
ngrams = icegrams.ngrams.Ngrams()

# Instantiate the BÍN database of inflectional forms
b = islenska.Bin()

# Read the data/nouns.csv file, which contains a list of noun lemmas.
# Look up each lemma in the BIN database (encapsulated in islenska),
# find all of its word forms, and look each wordform up in the icegrams
# database. Sum up the number of occurrences and store the lemma in
# the appropriate output bucket by frequency.

def file(name: str, mode: str) -> IO[str]:
    return open(Path("data") / name, mode, encoding="utf-8")

def bucket(freq: int) -> int:
    """ Return the bucket number for a given frequency, using powers of 10 """
    if freq <= 1:
        return 0
    # We only need buckets 0 through MAX_BUCKETS-1
    return min(MAX_BUCKETS - 1, int(math.log10(freq)))

class Buckets:

    """ A collection of frequency buckets for output lemmas """

    def __init__(self, name: str) -> None:
        self.name = name
        self.buckets: Dict[int, Set[str]] = defaultdict(set)
        self.lemmas: Dict[int, List[str]] = defaultdict(list)

    def add(self, lemma: str, freq: int) -> None:
        """ Add a lemma and its frequency to the appropriate bucket """
        # Find the bucket number
        b = bucket(freq)
        self.buckets[b].add(lemma)

    def write(self, limit: int) -> None:
        """ Sample the given number of lemmas from the buckets
            and write the samples to files, one file per bucket """
        for bucket, lemmas in self.buckets.items():
            if len(lemmas) < 1:
                continue
            with file(f"{self.name}-{bucket}.txt", "w") as out:
                llist = list(lemmas)
                samples = min(limit, len(llist))
                for lemma in random.sample(llist, samples):
                    out.write(f"{lemma}\n")

    def read(self, bucket: int) -> None:
        """ Read the contents of a bucket back from its file """
        try:
            bu = self.lemmas[bucket]
            with file(f"{self.name}-{bucket}.txt", "r") as inp:
                for line in inp:
                    if (line := line.strip()):
                        bu.append(line)
        except FileNotFoundError:
            # No lemmas found for this bucket
            assert len(self.lemmas[bucket]) == 0

    def choose(self, bucket: int) -> str:
        """ Choose a lemma at random from the specified bucket """
        if bucket not in self.lemmas:
            # Read the bucket from its file
            self.read(bucket)
        # Return a random lemma from the bucket
        return random.choice(self.lemmas[bucket])

def noun_generator() -> Iterator[NounTuple]:
    """ Return a generator to loop through the noun input file """
    with file("nouns.csv", "r") as inp:
        for line in inp:
            # Yield the noun lemma and the gender (kk, kvk, hk)
            if (line := line.strip()):
                n = line.split(",")
                if len(n) == 2:
                    lemma, gender = n[0], n[1]
                    if lemma[0].isupper():
                        # Skip proper nouns
                        continue
                    yield (lemma, gender)

def adjective_generator() -> Iterator[str]:
    """ Return a generator to loop through the adjective input file """
    with file("adjectives.csv", "r") as inp:
        for line in inp:
            # Yield the adjective lemma
            if (line := line.strip()):
                yield line

def process_nouns() -> None:
    """ Process noun lemmas """
    # Create the output buckets
    noun_buckets = Buckets("nouns")
    # Loop over the lemmas
    for lemma, gender in noun_generator():
        # Look up the lemma in the BIN database
        _, forms = b.lookup_lemmas(lemma)
        if len(forms) != 1 or forms[0].ofl != gender:
            # Skip nouns that have multiple lemmas/meanings
            continue

        _, forms = b.lookup(lemma)
        w = set(f.bmynd for f in forms if f.ord == lemma and f.ofl == gender)
        if any(any(e.ofl != gender or e.ord != lemma for e in b.lookup(wf)[1]) for wf in w):
            # Skip lemmas that are ambiguous, i.e. whose word forms can
            # belong to other lemmas
            continue

        # Loop over the distinct wordforms and sum their frequency
        freq = sum(ngrams.freq(wordform) for wordform in w)

        # Write the lemma and the frequency to the appropriate bucket
        noun_buckets.add(lemma, freq)

    # Done: write the result buckets to files
    noun_buckets.write(NUM_NOUN_SAMPLES)

def process_adjectives() -> None:
    """ Process adjective lemmas """
    adj_buckets = Buckets("adj")
    # Loop over the lemmas
    for lemma in adjective_generator():
        # Skip adjectives ending with "-legur" - they are
        # all the same for our purposes
        if lemma.endswith("legur"):
            continue
        if lemma in AVOID_ADJECTIVES:
            continue
        # Look up the lemma in the BIN database
        _, forms = b.lookup_lemmas(lemma)
        if len(forms) != 1 or forms[0].ofl != "lo":
            continue

        _, forms = b.lookup(lemma)
        w = set(f.bmynd for f in forms if f.ord == lemma and f.ofl == "lo")

        if any(any(e.ofl != "lo" or e.ord != lemma for e in b.lookup(wf)[1]) for wf in w):
            # Skip lemmas that are ambiguous, i.e. whose word forms can
            # belong to other lemmas
            continue

        # Loop over the distinct wordforms, summing up the frequencies
        freq = sum(ngrams.freq(wordform) for wordform in w)

        # Write the lemma and the frequency to the output file
        adj_buckets.add(lemma, freq)

    # Done: write the result buckets to files
    adj_buckets.write(NUM_ADJECTIVE_SAMPLES)

def generate(count: int) -> None:
    """ Generate JSONL output files """
    # Generate three output files, with varying degree of difficulty
    # from buckets 0, 1 and 2, by combining an adjective from bucket N
    # with a noun from bucket N.
    adj_buckets = Buckets("adj")
    noun_buckets = Buckets("nouns")
    jdump = functools.partial(json.dumps, ensure_ascii=False)
    for bucket in range(MAX_BUCKETS):
        # Read the adjective and noun buckets from file
        # Open the output file
        with file(f"icelandic-inflection-{DIFFICULTY[bucket]}.jsonl", "w") as out:
            c = 0
            while c < count:
                # Choose an adjective and a noun from the bucket
                adj_lemma = adj_buckets.choose(bucket)
                noun = noun_buckets.choose(bucket)
                gender = b.lookup(noun)[1][0].ofl
                # Find the base strong form of the adjective for the correct gender
                adj = b.lookup_variants(adj_lemma, "lo", (gender.upper(), "FSB", "NF", "ET"))[0].bmynd
                adj_ft = b.lookup_variants(adj_lemma, "lo", (gender.upper(), "FSB", "NF", "FT"))[0].bmynd
                try:
                    # Find the plural form of the noun
                    noun_ft = b.lookup_variants(noun, gender, ("NF", "FT"))[0].bmynd
                except IndexError:
                    # The noun probably does not exist in plural form
                    continue
                # Write the JSONL record to the output file
                # Create a noun phrase in singular and plural forms
                nl = NounPhrase(f"{adj} {noun}", force_number="et")
                nl_ft = NounPhrase(f"{adj_ft} {noun_ft}", force_number="ft")
                # Create the complete inflection JSON record
                completion = {
                    "et": {  # Singular
                        "nf": f"{nl:nf}",    # Nominative
                        "þf": f"{nl:þf}",    # Accusative
                        "þgf": f"{nl:þgf}",  # Dative
                        "ef": f"{nl:ef}"     # Genitive
                    },
                    "ft": {  # Plural
                        "nf": f"{nl_ft:nf}",
                        "þf": f"{nl_ft:þf}",
                        "þgf": f"{nl_ft:þgf}",
                        "ef": f"{nl_ft:ef}"
                    }
                }
                example = {
                    "input": [
                        {
                            "role": "system",
                            # "You are an expert in Icelandic grammar."
                            "content": "Þú ert sérfræðingur í íslenskri málfræði."
                        },
                        {
                            "role": "user",
                            "content": (
                                # "How does the noun phrase \"<adj> <noun>\" inflect in all cases (nf, þf, þgf, ef), "
                                # "singular (et) and plural (ft), without the definite article? "
                                # "Answer in JSON format only."
                                f"Hvernig fallbeygist nafnliðurinn \"{adj} {noun}\" í öllum föllum (nf, þf, þgf, ef), "
                                "eintölu (et) og fleirtölu (ft), án greinis? Svaraðu í JSON formi eingöngu."
                            )
                        }
                    ],
                    "ideal": jdump(completion)
                }
                out.write(f'{jdump(example)}\n')
                c += 1

if __name__ == "__main__":
    # Pick up command line arguments:
    # --nouns: process only nouns
    # --adjectives: process only adjectives
    # --all (default): process both nouns and adjectives
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nouns", action="store_true")
    parser.add_argument("--adjectives", action="store_true")
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()

    if args.generate:
        # Generate JSONL output files
        generate(10)  # 10 samples per bucket
    elif args.nouns:
        # Nouns only
        process_nouns()
    elif args.adjectives:
        # Adjectives only
        process_adjectives()
    else:
        # Process all
        process_nouns()
        process_adjectives()

