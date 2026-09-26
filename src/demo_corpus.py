"""离线演示语料生成器。

目的：不联网、不依赖任何外部下载，也能得到一个「规模不大但语义结构清晰」的语料，
让 tiny 配置在 CPU 上几分钟内训出可解释的词向量（近邻合理、类比能做对）。

做法：把人工撰写的自然句（data/samples/demo_corpus.txt）
      与按语义模板自动生成的事实句混合在一起。
      模板会把同一个事实用多种句式复述，模拟真实语料里「一个意思多种说法」的分布，
      同时刻意让不同类别共现（如 "The lion lives in the savanna and eats meat"），
      这样向量空间里才会形成可辨认的簇。
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List

from .utils import resolve_path

# --------------------------------------------------------------------------
# 语义资源
# --------------------------------------------------------------------------
CAPITAL_COUNTRY = [
    ("paris", "france", "french", "french"), ("berlin", "germany", "german", "german"),
    ("rome", "italy", "italian", "italian"), ("madrid", "spain", "spanish", "spanish"),
    ("tokyo", "japan", "japanese", "japanese"), ("moscow", "russia", "russian", "russian"),
    ("beijing", "china", "chinese", "chinese"), ("london", "england", "english", "english"),
    ("cairo", "egypt", "arabic", "egyptian"), ("athens", "greece", "greek", "greek"),
    ("lisbon", "portugal", "portuguese", "portuguese"), ("ottawa", "canada", "english", "canadian"),
    ("canberra", "australia", "english", "australian"), ("delhi", "india", "hindi", "indian"),
    ("seoul", "korea", "korean", "korean"), ("bangkok", "thailand", "thai", "thai"),
    ("vienna", "austria", "german", "austrian"), ("warsaw", "poland", "polish", "polish"),
    ("dublin", "ireland", "english", "irish"), ("oslo", "norway", "norwegian", "norwegian"),
    ("stockholm", "sweden", "swedish", "swedish"), ("helsinki", "finland", "finnish", "finnish"),
    ("brasilia", "brazil", "portuguese", "brazilian"), ("havana", "cuba", "spanish", "cuban"),
]

GENDER_PAIRS = [
    ("king", "queen", "royal"), ("man", "woman", "family"), ("boy", "girl", "family"),
    ("father", "mother", "family"), ("son", "daughter", "family"), ("brother", "sister", "family"),
    ("uncle", "aunt", "family"), ("nephew", "niece", "family"), ("husband", "wife", "family"),
    ("grandfather", "grandmother", "family"), ("prince", "princess", "royal"),
    ("emperor", "empress", "royal"), ("actor", "actress", "job"), ("waiter", "waitress", "job"),
    ("wizard", "witch", "story"), ("hero", "heroine", "story"), ("god", "goddess", "story"),
    ("gentleman", "lady", "social"),
]

COMPARATIVES = [
    ("big", "bigger"), ("small", "smaller"), ("tall", "taller"), ("short", "shorter"),
    ("old", "older"), ("new", "newer"), ("long", "longer"), ("fast", "faster"),
    ("slow", "slower"), ("warm", "warmer"), ("cold", "colder"), ("easy", "easier"),
    ("hard", "harder"), ("strong", "stronger"), ("bright", "brighter"), ("dark", "darker"),
    ("happy", "happier"), ("rich", "richer"), ("deep", "deeper"), ("high", "higher"),
    ("young", "younger"), ("cheap", "cheaper"), ("clean", "cleaner"), ("safe", "safer"),
]

OPPOSITES = [
    ("good", "bad"), ("hot", "cold"), ("light", "dark"), ("fast", "slow"),
    ("begin", "end"), ("open", "close"), ("up", "down"), ("young", "old"),
    ("rich", "poor"), ("strong", "weak"), ("tall", "short"), ("day", "night"),
    ("summer", "winter"), ("north", "south"), ("east", "west"), ("full", "empty"),
    ("early", "late"), ("wet", "dry"), ("heavy", "light"), ("noisy", "quiet"),
]

ANIMAL_YOUNG = [
    ("dog", "puppy"), ("cat", "kitten"), ("cow", "calf"), ("horse", "foal"),
    ("sheep", "lamb"), ("goat", "kid"), ("duck", "duckling"), ("lion", "cub"),
    ("bear", "cub"), ("frog", "tadpole"), ("hen", "chick"), ("pig", "piglet"),
    ("wolf", "cub"), ("deer", "fawn"),
]

ANIMAL_HABITAT = [
    ("lion", "savanna", "meat"), ("tiger", "jungle", "meat"), ("wolf", "forest", "meat"),
    ("whale", "ocean", "fish"), ("eagle", "mountain", "meat"), ("rabbit", "field", "grass"),
    ("fish", "river", "plants"), ("cow", "farm", "grass"), ("horse", "farm", "hay"),
    ("camel", "desert", "plants"), ("penguin", "ice", "fish"), ("monkey", "jungle", "fruit"),
    ("sheep", "meadow", "grass"), ("owl", "forest", "insects"), ("bee", "garden", "nectar"),
    ("shark", "ocean", "fish"), ("bear", "forest", "fish"), ("goat", "mountain", "grass"),
]

COLOR_OF = [
    ("apple", "red"), ("banana", "yellow"), ("grass", "green"), ("sky", "blue"),
    ("snow", "white"), ("coal", "black"), ("sun", "yellow"), ("rose", "red"),
    ("grape", "purple"), ("ocean", "blue"), ("leaf", "green"), ("blood", "red"),
    ("lemon", "yellow"), ("cloud", "white"), ("night", "black"), ("orange", "orange"),
]

JOB_VERB = [
    ("teacher", "teach", "school"), ("doctor", "heal", "hospital"), ("engineer", "build", "bridge"),
    ("writer", "write", "novel"), ("painter", "paint", "picture"), ("singer", "sing", "song"),
    ("farmer", "farm", "field"), ("driver", "drive", "car"), ("cook", "cook", "meal"),
    ("pilot", "fly", "airplane"), ("lawyer", "defend", "court"), ("scientist", "research", "lab"),
    ("programmer", "code", "software"), ("nurse", "care", "patient"), ("guard", "protect", "gate"),
]

FOOD_CATEGORY = [
    ("apple", "fruit"), ("banana", "fruit"), ("grape", "fruit"), ("orange", "fruit"),
    ("carrot", "vegetable"), ("potato", "vegetable"), ("onion", "vegetable"), ("cabbage", "vegetable"),
    ("bread", "food"), ("cheese", "food"), ("rice", "food"), ("fish", "food"),
    ("coffee", "drink"), ("tea", "drink"), ("water", "drink"), ("juice", "drink"),
]

TECH_WORDS = ["computer", "program", "code", "compiler", "model", "data", "server",
              "algorithm", "network", "memory", "gpu", "tensor", "vector", "training",
              "loss", "gradient", "batch", "token", "sentence", "corpus", "embedding",
              "dimension", "neighbor", "similarity", "analogy", "checkpoint", "epoch"]

NUMBER_WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight",
                "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
                "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]

VECTOR_WORDS = ["king", "queen", "man", "woman", "paris", "france", "dog", "cat",
                "computer", "model", "vector", "word", "sentence"]


# --------------------------------------------------------------------------
# 模板
# --------------------------------------------------------------------------
def _sentence_bank(rng: random.Random) -> List[str]:
    out: List[str] = []

    # ---- 首都 / 国家 / 语言 ----
    for cap, ctry, lang, demonym in CAPITAL_COUNTRY:
        out += [
            f"{cap} is the capital of {ctry} .",
            f"The capital of {ctry} is {cap} .",
            f"{cap} is a large city in {ctry} .",
            f"People who live in {ctry} are called {demonym} .",
            f"The {demonym} people live in {ctry} and many of them speak {lang} .",
            f"In {ctry} the people speak {lang} and the capital is {cap} .",
            f"She travelled from {cap} to another city in {ctry} by train .",
            f"{ctry} is a country and {cap} is its capital city .",
        ]

    # ---- 性别 / 家庭 ----
    for a, b, group in GENDER_PAIRS:
        out += [
            f"A {a} is a man and a {b} is a woman .",
            f"The {a} and the {b} walked into the room together .",
            f"A {b} is the female form of a {a} .",
            f"He is a {a} and she is a {b} .",
            f"The {a} ruled the kingdom and the {b} ruled beside him .",
        ]
        if group == "family":
            out += [
                f"My {a} visited my {b} in the city last summer .",
                f"The {a} and the {b} live in the same house .",
            ]
        if group == "royal":
            out += [
                f"The {a} and the {b} lived in a large palace .",
                f"Everyone in the kingdom obeyed the {a} and the {b} .",
            ]

    # ---- 比较级 ----
    for pos, comp in COMPARATIVES:
        out += [
            f"{comp.capitalize()} is the comparative form of {pos} .",
            f"This one is {comp} than that one .",
            f"A {comp} animal is more {pos} than a small one .",
            f"The road became {comp} as we walked further .",
            f"Today feels {comp} than yesterday .",
        ]

    # ---- 反义词 ----
    for a, b in OPPOSITES:
        out += [
            f"{a.capitalize()} and {b} are opposites .",
            f"The opposite of {a} is {b} and the opposite of {b} is {a} .",
            f"Something that is {a} is not {b} at all .",
        ]

    # ---- 动物幼崽 ----
    for adult, young in ANIMAL_YOUNG:
        out += [
            f"A {young} is a young {adult} .",
            f"The {adult} takes care of its {young} every day .",
            f"Every {adult} was once a small {young} .",
        ]

    # ---- 动物栖息地（跨类共现，帮助形成簇） ----
    for animal, habitat, food in ANIMAL_HABITAT:
        out += [
            f"The {animal} lives in the {habitat} and eats {food} .",
            f"You can find a {animal} in the {habitat} .",
            f"A {animal} is a wild animal that lives in the {habitat} .",
            f"The {habitat} is the natural home of the {animal} .",
        ]

    # ---- 颜色 ----
    for obj, color in COLOR_OF:
        out += [
            f"The {obj} is {color} .",
            f"A {obj} is usually {color} in color .",
            f"She painted the {obj} {color} .",
        ]

    # ---- 职业 ----
    for job, verb, place in JOB_VERB:
        out += [
            f"A {job} works in a {place} and {verb}s every day .",
            f"The {job} will {verb} the work before the deadline .",
            f"A {job} is a person whose job is to {verb} .",
            f"She works as a {job} in a {place} .",
        ]

    # ---- 食物分类 ----
    for item, cat in FOOD_CATEGORY:
        out += [
            f"An {item} is a kind of {cat} .",
            f"The {item} is a {cat} that people eat and drink every day .",
            f"He bought some {item} at the market .",
        ]

    # ---- 数字 ----
    for i, word in enumerate(NUMBER_WORDS, start=1):
        out += [
            f"The number {i} is written as {word} in words .",
            f"{word.capitalize()} is a number smaller than twenty .",
        ]
    for i in range(1, 11):
        out.append(f"{NUMBER_WORDS[i - 1].capitalize()} plus {NUMBER_WORDS[i - 1]} equals "
                   f"{NUMBER_WORDS[2 * i - 1]} .")

    # ---- 科技 / 词向量（让技术词聚成一簇） ----
    tech_templates = [
        "The {w1} processes the {w2} and writes a result .",
        "We store the {w1} in the {w2} before training .",
        "The {w1} is computed from the {w2} at every step .",
        "A good {w1} makes the {w2} run faster .",
        "The {w1} and the {w2} are both important for this task .",
        "He fixed the {w1} and the {w2} worked correctly again .",
    ]
    for tpl in tech_templates:
        for _ in range(14):
            w1, w2 = rng.sample(TECH_WORDS, 2)
            out.append(tpl.format(w1=w1, w2=w2))

    vector_templates = [
        "The word {w1} is closer to {w2} than to {w3} in the vector space .",
        "Word vectors place similar words like {w1} and {w2} close to each other .",
        "The vector of {w1} minus the vector of {w2} plus {w3} gives a meaningful word .",
        "The model learned that {w1} and {w2} appear in similar contexts .",
        "Cosine similarity between {w1} and {w2} is high because they are related .",
    ]
    for tpl in vector_templates:
        for _ in range(18):
            w1, w2, w3 = rng.sample(VECTOR_WORDS, 3)
            out.append(tpl.format(w1=w1, w2=w2, w3=w3))

    # ---- 通用自然句 ----
    generic = [
        "She read the book in the quiet library every afternoon .",
        "The children played football in the park on Sunday morning .",
        "He wrote a letter and posted it the following day .",
        "The music was loud and the room was full of people .",
        "The market sells fruit and vegetables every Saturday morning .",
        "The train arrived at the station ten minutes late .",
        "The airplane landed safely and the passengers left the plane .",
        "The ship crossed the ocean in seven days .",
        "A teacher explains the lesson and a student asks a question .",
        "The doctor examined the patient and wrote a prescription .",
        "The engineer designed the bridge and the workers built it .",
        "The farmer planted wheat in the field in early spring .",
        "The musician played the piano and the singer sang a song .",
        "Students learn faster when they see a clear example .",
        "The professor explained the algorithm on the whiteboard .",
        "The experiment produced a result that matched the theory .",
        "She presented her research at the conference in the summer .",
        "The paper described a simple method with strong results .",
        "The sun rises in the east and sets in the west .",
        "The moon orbits the earth and the earth orbits the sun .",
        "Gravity pulls objects toward the center of the earth .",
        "The heart pumps blood through the body every second .",
        "Doctors say that exercise keeps the body healthy .",
        "Sleep helps the brain store what you learned today .",
        "Water is a liquid and ice is a solid and steam is a gas .",
        "The winter was cold and the summer was hot and dry .",
        "It rained heavily yesterday and the sky is cloudy today .",
        "Spring brings warm weather and flowers bloom again .",
        "The library closes at eight and the cafe closes at ten .",
    ]
    out += generic
    return out


# --------------------------------------------------------------------------
# 对外接口
# --------------------------------------------------------------------------
def build_demo_corpus(seed: int = 42, extra_repeats: int = 6) -> str:
    """生成完整演示语料文本。"""
    rng = random.Random(seed)
    sentences = _sentence_bank(rng)

    # 把人工撰写的自然句也拼进来，增加真实句式
    hand = resolve_path("data/samples/demo_corpus.txt")
    if hand and hand.exists():
        with open(hand, "r", encoding="utf-8") as f:
            sentences += [ln.strip() for ln in f if ln.strip()]

    # 再整体打乱多轮，让同一事实的不同说法分散在语料各处
    corpus: List[str] = []
    for _ in range(extra_repeats):
        chunk = sentences[:]
        rng.shuffle(chunk)
        corpus += chunk
    rng.shuffle(corpus)
    return "\n".join(corpus) + "\n"


def write_demo_corpus(path, seed: int = 42) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = build_demo_corpus(seed=seed)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return len(text.split())


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="生成离线演示语料")
    ap.add_argument("--out", default="data/raw/demo/raw.txt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    n = write_demo_corpus(resolve_path(args.out), args.seed)
    print(f"已生成 {resolve_path(args.out)}，约 {n} 个词")
