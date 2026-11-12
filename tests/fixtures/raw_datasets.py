"""Tiny copies of the four raw datasets, in their real on-disk layouts."""

from __future__ import annotations

import zipfile
from pathlib import Path


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def liar_row(json_id: str, label: str, statement: str, speaker: str = "jane-doe") -> str:
    """One LIAR-PLUS line: index, id, label, statement, subject, speaker, job, state, party,
    five credit-history counts, context, justification (16 tab-separated columns)."""
    cols = ["0", json_id, label, statement, "taxes", speaker, "Senator", "Ohio", "democrat"]
    cols += ["1", "0", "2", "0", "0", "a speech", "The record shows otherwise."]
    return "\t".join(cols)


def make_raw_datasets(root: Path) -> Path:
    raw = root / "raw"
    _write(
        raw / "welfake" / "WELFake_Dataset.csv",
        ",title,text,label\n"
        '0,Senate passes bill,"WASHINGTON (Reuters) - The Senate voted on Tuesday.",0\n'
        "1,You won't believe this,Doctors hate this one trick that cures everything.,1\n"
        "2,Headline only,,1\n",
    )
    (raw / "isot").mkdir(parents=True)
    with zipfile.ZipFile(raw / "isot" / "News_dataset.zip", "w") as archive:
        archive.writestr(
            "Fake.csv",
            "title,text,subject,date\n"
            'Shocking truth,They lied to you about everything.,News,"December 31, 2017"\n',
        )
        archive.writestr(
            "True.csv",
            "title,text,subject,date\n"
            'Budget approved,"WASHINGTON (Reuters) - Lawmakers approved it.",politicsNews,'
            '"December 29, 2017 "\n',
        )
    fnn = raw / "fakenewsnet"
    header = "id,news_url,title,tweet_ids\n"
    _write(
        fnn / "politifact_fake.csv",
        header + "pf1,speedtalk.com/forum/viewtopic.php?t=1,Fake claim about a senator,1\t2\n",
    )
    _write(
        fnn / "politifact_real.csv",
        header + "pr1,https://www.nytimes.com/2017/a.html,Real report on the budget,3\n",
    )
    _write(
        fnn / "gossipcop_fake.csv",
        header + "gf1,www.dailymail.co.uk/tvshowbiz/x.html,Star secretly married,4\n",
    )
    _write(fnn / "gossipcop_real.csv", header + "gr1,,Star attends premiere,5\n")
    ratings = ["true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"]
    rows = [
        liar_row(f"{i}.json", rating, f"Statement number {i} about taxes.")
        for i, rating in enumerate(ratings)
    ]
    broken = "broken continuation line without tabs"
    _write(raw / "liar" / "train2.tsv", "\n".join(rows) + f"\n{broken}\n")
    _write(raw / "liar" / "val2.tsv", liar_row("90.json", "false", "Validation claim here.") + "\n")
    _write(raw / "liar" / "test2.tsv", liar_row("91.json", "true", "Test claim here.") + "\n")
    return raw
