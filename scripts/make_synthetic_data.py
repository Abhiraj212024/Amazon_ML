"""
Generate a synthetic dataset in the challenge's exact file layout.

This exists so the matching pipeline can be smoke-tested and its stages
compared without the real data. The noise operators mirror the patterns the
problem statement lists (abbreviations, legal-suffix drift, typos, token drops,
word-order transposition, landmark references, missing components), and the
generator deliberately produces the two hard cases:

  * singleton Source 1 entities with no match anywhere, and
  * Source 2/3 records for businesses that are absent from Source 1,

plus near-duplicate business names in the same city, which is what creates the
false merges the metric punishes.
"""
import argparse
import os
import random

import pandas as pd

NAME_HEADS = [
    "sunrise", "blue ocean", "golden gate", "silver leaf", "red fort", "green valley",
    "royal", "pioneer", "apex", "global", "united", "national", "premier", "quantum",
    "orchid", "lotus", "cedar", "maple", "granite", "falcon", "monsoon", "riverside",
]
NAME_TAILS = [
    "trading", "logistics", "textiles", "foods", "technologies", "motors", "pharma",
    "steel", "constructions", "enterprises", "solutions", "industries", "exports",
]
SUFFIXES = {
    "India": [("pvt ltd", "private limited"), ("ltd", "limited"), ("llp", "llp")],
    "US": [("inc", "incorporated"), ("llc", "l.l.c."), ("corp", "corporation")],
    "France": [("sarl", "s.a.r.l."), ("sas", "s.a.s."), ("sa", "societe anonyme")],
}
STREETS = {
    "India": ["mg road", "nehru street", "gandhi nagar", "ring road", "station road"],
    "US": ["main st", "oak ave", "sunset blvd", "park rd", "lincoln hwy"],
    "France": ["rue de rivoli", "avenue victor hugo", "boulevard saint germain"],
}
CITIES = {
    "India": ["mumbai", "pune", "chennai", "jaipur", "kochi"],
    "US": ["austin", "portland", "denver", "boston", "tampa"],
    "France": ["paris", "lyon", "nantes", "toulouse"],
}
LANDMARKS = ["near sbi atm", "opp city mall", "behind bus depot", "next to metro station"]
ABBREV = {"street": "st", "road": "rd", "avenue": "ave", "boulevard": "blvd", "highway": "hwy"}


def _typo(text, rng):
    if len(text) < 4:
        return text
    i = rng.randrange(1, len(text) - 1)
    kind = rng.random()
    if kind < 0.4:                                   # transpose
        return text[:i] + text[i + 1] + text[i] + text[i + 2:]
    if kind < 0.7:                                   # drop
        return text[:i] + text[i + 1:]
    return text[:i] + text[i] + text[i:]             # duplicate


def _noisy_name(name, suffix_pair, rng):
    short, long = suffix_pair
    parts = name.split()

    if rng.random() < 0.25:                          # word-order transposition
        rng.shuffle(parts)
    if rng.random() < 0.20 and len(parts) > 2:       # drop a token
        parts.pop(rng.randrange(len(parts)))

    text = " ".join(parts)
    if rng.random() < 0.30:
        text = _typo(text, rng)

    roll = rng.random()
    if roll < 0.35:
        text = f"{text} {short}"
    elif roll < 0.70:
        text = f"{text} {long}"
    if rng.random() < 0.15:
        text = text.replace(" and ", " & ")
    if rng.random() < 0.10:
        text = text.upper()
    return text


def _noisy_address(number, street, city, postcode, rng):
    text = street
    if rng.random() < 0.45:
        for full, abbr in ABBREV.items():
            text = text.replace(full, abbr)

    parts = [str(number), text, city]
    if rng.random() < 0.60:
        parts.append(postcode)                       # otherwise the PIN is missing
    if rng.random() < 0.20:
        parts.insert(rng.randrange(len(parts)), rng.choice(LANDMARKS))
    if rng.random() < 0.15:
        parts = parts[1:]                            # missing house number
    if rng.random() < 0.20:
        rng.shuffle(parts)

    address = ", ".join(parts)
    if rng.random() < 0.20:
        address = _typo(address, rng)
    return address


def build(n_entities, countries, singleton_rate, pool_only_rate, seed):
    rng = random.Random(seed)
    s1, s2, s3, ground_truth = [], [], [], []
    counters = {"S1": 0, "S2": 0, "S3": 0}

    def next_id(prefix):
        counters[prefix] += 1
        return f"{prefix}-{counters[prefix]:05d}"

    n_total = int(n_entities * (1 + pool_only_rate))

    for i in range(n_total):
        country = countries[i % len(countries)]
        head, tail = rng.choice(NAME_HEADS), rng.choice(NAME_TAILS)
        # a shared head within a city is what produces genuinely confusable pairs
        base_name = f"{head} {tail}"
        suffix_pair = rng.choice(SUFFIXES[country])
        number = rng.randrange(1, 900)
        street = rng.choice(STREETS[country])
        city = rng.choice(CITIES[country])
        postcode = f"{rng.randrange(10000, 99999)}"

        in_source1 = i < n_entities
        is_singleton = in_source1 and rng.random() < singleton_rate

        if in_source1:
            s1_id = next_id("S1")
            s1.append({
                "entity_id": s1_id,
                "business_name": _noisy_name(base_name, suffix_pair, rng),
                "business_address": _noisy_address(number, street, city, postcode, rng),
                "country": country,
            })

        matched = []
        if not (in_source1 and is_singleton):
            n_s2 = rng.choices([0, 1, 2, 3], weights=[0.25, 0.45, 0.20, 0.10])[0]
            n_s3 = rng.choices([0, 1, 2], weights=[0.40, 0.45, 0.15])[0]
            for prefix, count, bucket in (("S2", n_s2, s2), ("S3", n_s3, s3)):
                for _ in range(count):
                    rid = next_id(prefix)
                    bucket.append({
                        "entity_id": rid,
                        "business_name": _noisy_name(base_name, suffix_pair, rng),
                        "business_address": _noisy_address(number, street, city, postcode, rng),
                        "country": country,
                    })
                    if in_source1:
                        matched.append(rid)

        if in_source1:
            ground_truth.append({
                "source1_entity_id": s1_id,
                "matched_entity_ids": ",".join(matched),
            })

    return (pd.DataFrame(s1), pd.DataFrame(s2), pd.DataFrame(s3),
            pd.DataFrame(ground_truth))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="dataset_synthetic")
    parser.add_argument("--n-train", type=int, default=1200)
    parser.add_argument("--n-test", type=int, default=400)
    parser.add_argument("--singleton-rate", type=float, default=0.30)
    parser.add_argument("--pool-only-rate", type=float, default=0.35,
                        help="extra businesses present only in Source 2/3, as distractors")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    train_dir = os.path.join(args.output_dir, "train")
    test_dir = os.path.join(args.output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    s1, s2, s3, gt = build(
        args.n_train, ["India", "US"], args.singleton_rate, args.pool_only_rate, args.seed
    )
    s1.to_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t", index=False)
    s2.to_csv(os.path.join(train_dir, "train_source2.tsv"), sep="\t", index=False)
    s3.to_csv(os.path.join(train_dir, "train_source3.tsv"), sep="\t", index=False)
    gt.to_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t", index=False)

    # the test split adds France, which never appears in training
    t1, t2, t3, tgt = build(
        args.n_test, ["India", "US", "France"], args.singleton_rate,
        args.pool_only_rate, args.seed + 1,
    )
    t1.to_csv(os.path.join(test_dir, "test_source1.tsv"), sep="\t", index=False)
    t2.to_csv(os.path.join(test_dir, "test_source2.tsv"), sep="\t", index=False)
    t3.to_csv(os.path.join(test_dir, "test_source3.tsv"), sep="\t", index=False)
    # kept out of the challenge layout on purpose; only for local scoring
    tgt.to_csv(os.path.join(test_dir, "test_ground_truth_HIDDEN.tsv"), sep="\t", index=False)

    print(f"train: s1={len(s1)} s2={len(s2)} s3={len(s3)}")
    print(f"test:  s1={len(t1)} s2={len(t2)} s3={len(t3)}")
    print(f"written to {args.output_dir}/")


if __name__ == "__main__":
    main()
